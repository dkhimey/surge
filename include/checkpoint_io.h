#pragma once
//
// checkpoint_io.h — durability + integrity helpers for checkpoint writes.
//
// Background: the checkpoint code used to write shard / metadata files with a
// plain std::ofstream and rely on close() for durability. close() only hands
// the bytes to the OS page cache; it does NOT force them to stable storage.
// If the job is later SIGKILL'd (scheduler time limit / OOM) or the node
// crashes, the large shard files can still be sitting in the page cache and are
// lost, while small files written earlier (e.g. results.csv) have already been
// flushed. On the next run the resume picks a checkpoint whose shard files are
// truncated, and hnswlib's loader aborts with
//     "Index seems to be corrupted or unsupported".
//
// These helpers provide:
//   * fsync_file / fsync_dir / fsync_parent_dir — force data + directory
//     entries to disk.
//   * atomic_durable_write — write-to-temp + fsync + rename + dir-fsync, so a
//     published file is always either the complete old version or the complete
//     new one, never a torn mix.
//   * hnsw_file_is_valid — a cheap, allocation-free structural check that an
//     hnswlib index file on disk is complete. It mirrors the integrity walk in
//     HierarchicalNSW::loadIndex but does NOT allocate the graph, so it can be
//     run on a freshly-written 500M-scale shard without doubling memory. This
//     catches silent short writes (e.g. ENOSPC) that saveIndex ignores, before
//     the checkpoint is committed.
//
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include <fcntl.h>
#include <unistd.h>

namespace surge {

// fsync a regular file by path. Returns true on success.
inline bool fsync_file(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return false;
    int r = ::fsync(fd);
    ::close(fd);
    return r == 0;
}

// fsync a directory so that newly created/renamed entries are durable.
// Best-effort: some filesystems reject directory fsync (returns false), which
// callers may treat as non-fatal.
inline bool fsync_dir(const std::string& dir) {
    int flags = O_RDONLY;
#ifdef O_DIRECTORY
    flags |= O_DIRECTORY;
#endif
    int fd = ::open(dir.c_str(), flags);
    if (fd < 0) return false;
    int r = ::fsync(fd);
    ::close(fd);
    return r == 0;
}

inline bool fsync_parent_dir(const std::string& path) {
    std::filesystem::path p(path);
    std::filesystem::path parent = p.parent_path();
    if (parent.empty()) parent = ".";
    return fsync_dir(parent.string());
}

// Durably and atomically write `bytes` to `final_path`:
//   write to final_path + ".tmp" -> flush -> fsync -> rename -> fsync(dir).
// Throws std::runtime_error on any failure. Suitable for small files
// (metadata, commit markers). Do NOT use for multi-GB buffers.
inline void atomic_durable_write(const std::string& final_path,
                                 const std::string& bytes) {
    const std::string tmp = final_path + ".tmp";
    {
        std::ofstream o(tmp, std::ios::binary | std::ios::trunc);
        if (!o) throw std::runtime_error("cannot open for write: " + tmp);
        if (!bytes.empty())
            o.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        o.flush();
        if (!o) throw std::runtime_error("write failed (disk full?): " + tmp);
    }
    if (!fsync_file(tmp))
        throw std::runtime_error("fsync failed: " + tmp);
    std::error_code ec;
    std::filesystem::rename(tmp, final_path, ec);
    if (ec)
        throw std::runtime_error("rename failed: " + final_path + ": " + ec.message());
    fsync_parent_dir(final_path);  // best-effort; rename durability
}

// Structural integrity check for an hnswlib HierarchicalNSW<float> file.
//
// Mirrors the "check if index is ok" walk inside HierarchicalNSW::loadIndex
// (external/hnswlib/hnswlib/hnswalg.h): read the fixed header, skip the level-0
// data block, then walk one variable-length link list per element and confirm
// the cursor lands exactly on EOF. Returns false (with an explanation in `err`,
// if provided) when the file is truncated, short-written, or otherwise the size
// the header implies does not match the bytes on disk — i.e. exactly the
// condition that makes loadIndex throw "Index seems to be corrupted".
//
// Allocation-free: only seeks through the file, never materialises the graph.
inline bool hnsw_file_is_valid(const std::string& path, std::string* err = nullptr) {
    auto fail = [&](const std::string& m) {
        if (err) *err = m;
        return false;
    };

    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return fail("cannot open file");

    in.seekg(0, std::ios::end);
    const std::streamoff filesize = in.tellg();
    if (filesize < 0) return fail("cannot determine file size");
    in.seekg(0, std::ios::beg);

    auto rd = [&](void* dst, size_t n) -> bool {
        in.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(n));
        return static_cast<bool>(in) && in.gcount() == static_cast<std::streamsize>(n);
    };

    // Header fields, in the exact order/sizes saveIndex writes them.
    uint64_t offsetLevel0, max_elements, cur_element_count, size_data_per_element,
             label_offset, offsetData, maxM, maxM0, M, ef_construction;
    int32_t  maxlevel;
    uint32_t enterpoint_node;
    double   mult;

    if (!rd(&offsetLevel0, 8)          || !rd(&max_elements, 8) ||
        !rd(&cur_element_count, 8)      || !rd(&size_data_per_element, 8) ||
        !rd(&label_offset, 8)          || !rd(&offsetData, 8) ||
        !rd(&maxlevel, 4)              || !rd(&enterpoint_node, 4) ||
        !rd(&maxM, 8)                  || !rd(&maxM0, 8) ||
        !rd(&M, 8)                     || !rd(&mult, 8) ||
        !rd(&ef_construction, 8)) {
        return fail("header truncated");
    }
    (void)offsetLevel0; (void)max_elements; (void)label_offset; (void)offsetData;
    (void)maxlevel; (void)enterpoint_node; (void)maxM; (void)maxM0; (void)M;
    (void)mult; (void)ef_construction;

    if (size_data_per_element == 0)
        return fail("invalid header: size_data_per_element == 0");

    const std::streamoff header_end = in.tellg();

    // Skip the contiguous level-0 data block. Guard against a header that
    // claims more data than the file can hold (long double avoids overflow in
    // the comparison even for huge element counts).
    const long double data_block =
        static_cast<long double>(cur_element_count) *
        static_cast<long double>(size_data_per_element);
    if (static_cast<long double>(header_end) + data_block >
        static_cast<long double>(filesize)) {
        return fail("truncated: level-0 data block extends past EOF");
    }
    in.seekg(header_end +
                 static_cast<std::streamoff>(cur_element_count * size_data_per_element),
             std::ios::beg);

    // Walk one linkListSize-prefixed list per element.
    for (uint64_t i = 0; i < cur_element_count; ++i) {
        const std::streamoff cur = in.tellg();
        if (cur < 0 || cur >= filesize)
            return fail("truncated: link-list region overruns EOF");
        uint32_t link_list_size = 0;
        if (!rd(&link_list_size, 4))
            return fail("truncated: cannot read linkListSize");
        if (link_list_size != 0)
            in.seekg(static_cast<std::streamoff>(link_list_size), std::ios::cur);
    }

    const std::streamoff end_pos = in.tellg();
    if (end_pos != filesize)
        return fail("size mismatch: trailing/missing bytes (corrupted or partial write)");

    return true;
}

}  // namespace surge
