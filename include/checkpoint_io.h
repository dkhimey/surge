#pragma once
//
// checkpoint_io.h — durability and integrity helpers for checkpoint writes.
//
// Provides atomic writes with fsync, directory durability, and structural
// validation for hnswlib index files to prevent corruption from partial writes.
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

// fsync a regular file by path, returns true on success
inline bool fsync_file(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return false;
    int r = ::fsync(fd);
    ::close(fd);
    return r == 0;
}

// fsync a directory so entries are durable; best-effort (some filesystems reject it)
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

// Atomically write to final_path via tmp file + fsync + rename
// Throws std::runtime_error on failure; unsuitable for multi-GB buffers
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
    fsync_parent_dir(final_path);  // best-effort durability
}

// Structural validation of hnswlib index files without loading graph
// Mirrors HierarchicalNSW::loadIndex validation but allocation-free
// Returns false if truncated, short-written, or size mismatches occur
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

    // Read header fields in saveIndex order
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

    // Skip level-0 data block; guard against overflow with long double
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

    // Validate link lists for each element
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
