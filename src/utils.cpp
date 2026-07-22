// utils.cpp
//
// Dataset registry (DATASETS) and shared I/O helpers: file-format detection,
// vector and ground-truth readers, and sampling.

#include "utils.h"

// ─────────────────────────────────────────────────────────────────────────────
// DATASETS registry (declared extern in utils.h)
//
// Two kinds of entries:
//   • Streaming (dynamic) datasets — fields: base_file, runbook, query_file,
//     ground_truth_dir.  One per (family, scale, workload).
//   • Static (base) datasets       — fields: base_file, query_file, gt_file.
//
// Families: bigann/sift (u8, 128-dim) and msturing (f32, 100-dim).
// Edit the paths below to point at your local copies of the data.
// ─────────────────────────────────────────────────────────────────────────────
std::unordered_map<std::string, std::map<std::string, std::string>> DATASETS = {

    // ── Streaming datasets: BIGANN ───────────────────────────────────────────
    {"bigann-100M-clustered",
        {
            {"base_file",        "/dataset/big-ann-benchmarks/data/bigann-clustered/bigann-100M-clustered.u8bin"},
            {"runbook",          "/dataset/big-ann-benchmarks/data/bigann-clustered/runbook-bigann-100M.yaml"},
            {"query_file",       "/dataset/big-ann-benchmarks/data/bigann-clustered/query.public.10K.u8bin"},
            {"ground_truth_dir", "/dataset/big-ann-benchmarks/data/bigann-clustered/100000000/runbook-bigann-100M.yaml"},
        }
    },
    {"bigann-100M-random",
        {
            {"base_file",        "/dataset/big-ann-benchmarks/data/bigann-random/bigann-100M-random.u8bin"},
            {"runbook",          "/dataset/big-ann-benchmarks/data/bigann-random/runbook-bigann-100M-random.yaml"},
            {"query_file",       "/dataset/big-ann-benchmarks/data/bigann-random/query.public.10K.u8bin"},
            {"ground_truth_dir", "/dataset/big-ann-benchmarks/data/bigann-random/100000000/runbook-bigann-100M-random.yaml"},
        }
    },
    {"bigann-100M-shift",
        {
            {"base_file",        "/dataset/big-ann-benchmarks/data/bigann-shift/100M-bigann-shift.u8bin"},
            {"runbook",          "/dataset/big-ann-benchmarks/data/bigann-shift/bigann-100M-shift_runbookfinal.yaml"},
            {"query_file",       "/dataset/big-ann-benchmarks/data/bigann-shift/query.public.10K.u8bin"},
            {"ground_truth_dir", "/dataset/big-ann-benchmarks/data/bigann-shift/100000000/bigann-100M-shift_runbookfinal.yaml"},
        }
    },
    {"bigann-500M-clustered",
        {
            {"base_file",        "/dataset/big-ann-benchmarks/data/bigann-500M-clustered/500M-bigann64clustered.u8bin"},
            {"runbook",          "/dataset/big-ann-benchmarks/data/bigann-500M-clustered/runbook_bigann-500M-clustered.yaml"},
            {"query_file",       "/dataset/big-ann-benchmarks/data/bigann-500M-clustered/query.public.10K.u8bin"},
            {"ground_truth_dir", "/dataset/big-ann-benchmarks/data/bigann-500M-clustered/500000000/runbook_bigann-500M-clustered.yaml/"},
        }
    },
    {"bigann-500M-shift",
        {
            {"base_file",        "/dataset/big-ann-benchmarks/data/bigann-500M-shift/500M_bigann500shift64.u8bin"},
            {"runbook",          "/dataset/big-ann-benchmarks/data/bigann-500M-shift/runbook_bigann500Mshift.yaml"},
            {"query_file",       "/dataset/big-ann-benchmarks/data/bigann-500M-shift/query.public.10K.u8bin"},
            {"ground_truth_dir", "/dataset/big-ann-benchmarks/data/bigann-500M-shift/500000000/runbook_bigann500Mshift.yaml/"},
        }
    },

    // ── Streaming datasets: MSTuring ─────────────────────────────────────────
    {"msturing-100M-clustered",
        {
            {"base_file",        "/dataset/big-ann-benchmarks/data/MSTuring-100M-clustered/100M-msturing-clustered.fbin"},
            {"runbook",          "/dataset/big-ann-benchmarks/data/MSTuring-100M-clustered/msturing-100M-clustered_runbookfinal.yaml"},
            {"query_file",       "/dataset/big-ann-benchmarks/data/MSTuring-100M-clustered/testQuery10K.fbin"},
            {"ground_truth_dir", "/dataset/big-ann-benchmarks/data/MSTuring-100M-clustered/100000000/msturing-100M-clustered_runbookfinal.yaml"},
        }
    },
    {"msturing-100M-random",
        {
            {"base_file",        "/dataset/big-ann-benchmarks/data/MSTuring-100M-random/msturing-100M-random.fbin"},
            {"runbook",          "/dataset/big-ann-benchmarks/data/MSTuring-100M-random/runbook-msturing-100M-random.yaml"},
            {"query_file",       "/dataset/big-ann-benchmarks/data/MSTuring-100M-random/testQuery10K.fbin"},
            {"ground_truth_dir", "/dataset/big-ann-benchmarks/data/MSTuring-100M-random/100000000/runbook-msturing-100M-random.yaml"},
        }
    },
    {"msturing-100M-shift",
        {
            {"base_file",        "/dataset/big-ann-benchmarks/data/MSTuring-100M-shift/100M-msturing-shift.fbin"},
            {"runbook",          "/dataset/big-ann-benchmarks/data/MSTuring-100M-shift/msturing-100M-shift_runbookfinal.yaml"},
            {"query_file",       "/dataset/big-ann-benchmarks/data/MSTuring-100M-shift/testQuery10K.fbin"},
            {"ground_truth_dir", "/dataset/big-ann-benchmarks/data/MSTuring-100M-shift/100000000/msturing-100M-shift_runbookfinal.yaml"},
        }
    },
    {"msturing-500M-clustered",
        {
            {"base_file",        "/dataset/big-ann-benchmarks/data/MSTuring-500M-clustered/500M-msturingclustered64.fbin"},
            {"runbook",          "/dataset/big-ann-benchmarks/data/MSTuring-500M-clustered/runbook_msturing500Mclustered.yaml"},
            {"query_file",       "/dataset/big-ann-benchmarks/data/MSTuring-500M-clustered/testQuery10K.fbin"},
            {"ground_truth_dir", "/dataset/big-ann-benchmarks/data/MSTuring-500M-clustered/500000000/runbook_msturing500Mclustered.yaml"},
        }
    },
    {"msturing-500M-shift",
        {
            {"base_file",        "/dataset/big-ann-benchmarks/data/MSTuring-500M-shift/500M_msturingshift64.fbin"},
            {"runbook",          "/dataset/big-ann-benchmarks/data/MSTuring-500M-shift/runbook_msturing500Mshift.yaml"},
            {"query_file",       "/dataset/big-ann-benchmarks/data/MSTuring-500M-shift/testQuery10K.fbin"},
            {"ground_truth_dir", "/dataset/big-ann-benchmarks/data/MSTuring-500M-shift/500000000/runbook_msturing500Mshift.yaml"},
        }
    },

    // ── Static datasets: SIFT / BIGANN ───────────────────────────────────────
    {"bigann-100M",
        {
            {"base_file",  "/dataset/big-ann-benchmarks/data/bigann/base.1B.crop_nb_100000000.u8bin"},
            {"query_file", "/dataset/big-ann-benchmarks/data/bigann/query.public.10K.u8bin"},
            {"gt_file",    "/dataset/big-ann-benchmarks/data/bigann/bigann-100M.ibin"},
        }
    },
    {"bigann-500M",
        {
            {"base_file",  "/dataset/big-ann-benchmarks/data/bigann/base.1B.crop_nb_500000000.u8bin"},
            {"query_file", "/dataset/big-ann-benchmarks/data/bigann/query.public.10K.u8bin"},
            {"gt_file",    "/dataset/big-ann-benchmarks/data/bigann/gt_computed.10.crop500M.ibin"},
        }
    },
    {"bigann-1B",
        {
            {"base_file",  "/dataset/big-ann-benchmarks/data/bigann/base.1B.u8bin"},
            {"query_file", "/dataset/big-ann-benchmarks/data/bigann/query.public.10K.u8bin"},
            {"gt_file",    "/dataset/big-ann-benchmarks/data/bigann/GT.public.1B.ibin"},
        }
    },

    // ── Static datasets: MSTuring ────────────────────────────────────────────
    {"msturing-100M",
        {
            {"base_file",  "/dataset/big-ann-benchmarks/data/MSTuringANNS/base1b.crop_nb_100000000.fbin"},
            {"query_file", "/dataset/big-ann-benchmarks/data/MSTuringANNS/query10K.fbin"},
            {"gt_file",    "/dataset/big-ann-benchmarks/data/MSTuringANNS/msturing-gt-100M.query10K.bin"},
        }
    },
    {"msturing-500M",
        {
            {"base_file",  "/dataset/big-ann-benchmarks/data/MSTuringANNS/base1b.crop_nb_500000000.fbin"},
            {"query_file", "/dataset/big-ann-benchmarks/data/MSTuringANNS/testQuery10K.fbin"},
            {"gt_file",    "/dataset/big-ann-benchmarks/data/MSTuringANNS/gt_computed.10.crop_500M.ibin"},
        }
    },
    {"msturing-1B",
        {
            {"base_file",  "/dataset/big-ann-benchmarks/data/MSTuringANNS/base1b.fbin"},
            {"query_file", "/dataset/big-ann-benchmarks/data/MSTuringANNS/testQuery10K.fbin"},
            {"gt_file",    "/dataset/big-ann-benchmarks/data/MSTuringANNS/gt_computed.10.full_1B.ibin"},
        }
    },
};

FileFormat getFileFormat(const std::string& filename) {
    if (filename.size() >= 6 && filename.substr(filename.size() - 6) == ".fvecs") {
        return FVECS;
    } else if (filename.size() >= 6 && filename.substr(filename.size() - 6) == ".bvecs") {
        return BVECS;
    } else if (filename.size() >= 6 && filename.substr(filename.size() - 6) == ".i8bin") {
        return I8BIN;
    } else if (filename.size() >= 6 && filename.substr(filename.size() - 6) == ".u8bin") {
        return U8BIN;
    } else if (filename.size() >= 5 && filename.substr(filename.size() - 5) == ".fbin") {
        return FBIN;
    } else {
        std::cerr << "Unsupported file format for file: " << filename << "\n";
        exit(1);
    }
}

std::pair<int, int> get_dataset_info(const std::string& base_file) {
    std::ifstream file(base_file, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "FATAL: cannot open dataset file: " << base_file << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    FileFormat format = getFileFormat(base_file);
    switch (format) {
        case BVECS:
        case FVECS: {
            uint32_t num_vectors, dim;
            file.read(reinterpret_cast<char*>(&num_vectors), sizeof(uint32_t));
            file.read(reinterpret_cast<char*>(&dim), sizeof(uint32_t));
            return {num_vectors, dim};
        }
        case I8BIN:
        case U8BIN:
        case FBIN: {
            uint32_t num_vectors, dim;
            file.read(reinterpret_cast<char*>(&num_vectors), sizeof(uint32_t));
            file.read(reinterpret_cast<char*>(&dim), sizeof(uint32_t));
            return {num_vectors, dim};
        }
        default:
            std::cerr << "Unsupported file format " << base_file << "\n";
            exit(1);
    }
}

std::vector<float> getSample(const std::string& filename, size_t max_elements, size_t dim, size_t sample_size, size_t start_offset) {
    std::mt19937 gen(std::random_device{}());
    FileFormat filetype = getFileFormat(filename);

    std::vector<int> indices(max_elements);
    std::iota(indices.begin(), indices.end(), static_cast<int>(start_offset));

    std::vector<int> sampled_indices;
    std::sample(indices.begin(), indices.end(), std::back_inserter(sampled_indices),
                sample_size, gen);

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for sampling: " << filename << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::vector<float> sample;
    sample.reserve(sample_size * dim);

    size_t header_offset = 0;

    std::pair<int,int> data_info = get_dataset_info(filename);
    int header_num_pts = data_info.first;
    int header_dim = data_info.second;

    if (filetype == I8BIN || filetype == U8BIN || filetype == FBIN) {
        header_offset = 8;
    }

    for (int idx : sampled_indices) {
        switch (filetype) {
            case FVECS: {
                size_t offset = idx * (sizeof(int) + sizeof(float) * dim);
                file.seekg(offset, std::ios::beg);

                int d;
                file.read(reinterpret_cast<char*>(&d), sizeof(int));
                if (d != static_cast<int>(dim)) {
                    std::cerr << "Fvecs dim mismatch at index " << idx << ": " << d << " != " << dim << "\n";
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }

                std::vector<float> vec(dim);
                file.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(float));
                sample.insert(sample.end(), vec.begin(), vec.end());
                break;
            }
            case BVECS: {
                size_t offset = idx * (sizeof(int) + sizeof(uint8_t) * dim);
                file.seekg(offset, std::ios::beg);

                int d;
                file.read(reinterpret_cast<char*>(&d), sizeof(int));
                if (d != static_cast<int>(dim)) {
                    std::cerr << "Bvecs dim mismatch at index " << idx << ": " << d << " != " << dim << "\n";
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }

                std::vector<uint8_t> vec(dim);
                file.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(uint8_t));
                for (uint8_t v : vec) sample.push_back(static_cast<float>(v));
                break;
            }
            case I8BIN: {
                size_t offset = header_offset + static_cast<size_t>(idx) * dim * sizeof(int8_t);
                file.seekg(offset, std::ios::beg);

                std::vector<int8_t> vec(dim);
                file.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(int8_t));
                for (int8_t v : vec) sample.push_back(static_cast<float>(v));
                break;
            }
            case U8BIN: {
                size_t offset = header_offset + static_cast<size_t>(idx) * dim * sizeof(uint8_t);
                file.seekg(offset, std::ios::beg);

                std::vector<uint8_t> vec(dim);
                file.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(uint8_t));
                for (uint8_t v : vec) sample.push_back(static_cast<float>(v));
                break;
            }
            case FBIN: {
                size_t offset = header_offset + static_cast<size_t>(idx) * dim * sizeof(float);
                file.seekg(offset, std::ios::beg);

                std::vector<float> vec(dim);
                file.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(float));
                if (!file) {
                    std::cerr << "Failed to read float vector at index " << idx << "\n";
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }

                sample.insert(sample.end(), vec.begin(), vec.end());
                break;
            }
            default:
                std::cerr << "Unsupported file format " << filename << "\n";
                MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    return sample;
}

std::vector<float> readFvecs(const std::string& filename, size_t vector_dim, int n, int offset) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[Coordinator]: Unable to open Fvecs file\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
        return std::vector<float>();
    }

    size_t bytes_per_vector = 4 + vector_dim * 4;
    size_t byte_offset = static_cast<size_t>(offset) * bytes_per_vector;

    file.seekg(byte_offset, std::ios::beg);
    if (!file.good()) {
        std::cerr << "[Coordinator]: Failed to seek to offset in fvecs file\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
        return std::vector<float>();
    }

    std::vector<float> vectors;
    if (n != INT_MAX)
        vectors.reserve(static_cast<size_t>(n) * vector_dim);

    for (int counter = 0; counter < n && file; ++counter) {
        int dim;
        file.read(reinterpret_cast<char*>(&dim), sizeof(int));
        if (file.eof()) break;

        if (dim != static_cast<int>(vector_dim)) {
            std::cerr << "[Coordinator]: Dimension mismatch at vector " << offset + counter
                      << ": file " << filename << " says " << dim << ", expected " << vector_dim << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
            return std::vector<float>();
        }

        std::vector<float> temp(vector_dim);
        file.read(reinterpret_cast<char*>(temp.data()), vector_dim * sizeof(float));
        if (file.gcount() != static_cast<std::streamsize>(vector_dim * sizeof(float))) {
            std::cerr << "[Coordinator]: Error reading vector data at position " << offset + counter << "\n";
            break;
        }

        vectors.insert(vectors.end(), temp.begin(), temp.end());
    }

    return vectors;
}

std::vector<float> readBvecs(const std::string& filename, size_t vector_dim, int n, int offset) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[Coordinator]: Unable to open Bvecs file\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
        return std::vector<float>();
    }

    size_t bytes_per_vector = sizeof(int) + vector_dim * sizeof(uint8_t);
    size_t byte_offset = static_cast<size_t>(offset) * bytes_per_vector;

    file.seekg(byte_offset, std::ios::beg);
    if (!file.good()) {
        std::cerr << "[Coordinator]: Failed to seek to offset in bvecs file\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
        return std::vector<float>();
    }

    std::vector<float> vectors;
    if (n != INT_MAX)
        vectors.reserve(static_cast<size_t>(n) * vector_dim);

    for (int counter = 0; counter < n && file; ++counter) {
        int dim;
        file.read(reinterpret_cast<char*>(&dim), sizeof(int));
        if (file.eof()) break;

        if (dim != static_cast<int>(vector_dim)) {
            std::cerr << "[Coordinator]: Dimension mismatch at vector " << offset + counter
                      << ": file says " << dim << ", expected " << vector_dim << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
            return std::vector<float>();
        }

        std::vector<uint8_t> temp_bytes(vector_dim);
        file.read(reinterpret_cast<char*>(temp_bytes.data()), vector_dim * sizeof(uint8_t));
        if (file.gcount() != static_cast<std::streamsize>(vector_dim * sizeof(uint8_t))) {
            std::cerr << "[Coordinator]: Error reading vector data at position " << offset + counter << "\n";
            break;
        }

        for (uint8_t val : temp_bytes) {
            vectors.push_back(static_cast<float>(val));
        }
    }

    return vectors;
}

std::vector<float> readI8bin(const std::string& filename, size_t vector_dim, int n, int offset) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[Coordinator]: Unable to open i8bin file " << filename << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
        return {};
    }

    uint32_t num_points, dim_in_file;
    file.read(reinterpret_cast<char*>(&num_points), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&dim_in_file), sizeof(uint32_t));

    if (dim_in_file != vector_dim) {
        std::cerr << "[Coordinator]: Dimension mismatch: file "<< filename << " says " << dim_in_file
                  << ", expected " << vector_dim << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
        return {};
    }

    if (offset < 0 || static_cast<size_t>(offset) >= num_points) {
        std::cerr << "[Coordinator]: Invalid offset\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
        return {};
    }

    size_t max_readable = num_points - static_cast<size_t>(offset);
    size_t num_to_read = std::min(static_cast<size_t>(n), max_readable);

    std::vector<int8_t> buffer(num_to_read * vector_dim);
    size_t offset_bytes = static_cast<size_t>(offset) * vector_dim * sizeof(int8_t);
    file.seekg(8 + offset_bytes, std::ios::beg);

    file.read(reinterpret_cast<char*>(buffer.data()), buffer.size() * sizeof(int8_t));
    if (file.gcount() != static_cast<std::streamsize>(buffer.size() * sizeof(int8_t))) {
        std::cerr << "[Coordinator]: Failed to read the full data block\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
        return {};
    }

    std::vector<float> vectors;
    vectors.reserve(num_to_read * vector_dim);
    for (int8_t val : buffer) {
        vectors.push_back(static_cast<float>(val));
    }

    return vectors;
}

std::vector<float> readU8bin(const std::string& filename, size_t vector_dim, int n, int offset) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[Coordinator]: Unable to open u8bin file " << filename << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
        return {};
    }

    uint32_t num_points, dim_in_file;
    file.read(reinterpret_cast<char*>(&num_points), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&dim_in_file), sizeof(uint32_t));

    if (dim_in_file != vector_dim) {
        std::cerr << "[Coordinator]: Dimension mismatch: file " << filename << " says " << dim_in_file
                  << ", expected " << vector_dim << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
        return {};
    }

    if (offset < 0 || static_cast<size_t>(offset) >= num_points) {
        std::cerr << "[Coordinator]: Invalid offset\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
        return {};
    }

    size_t max_readable = num_points - static_cast<size_t>(offset);
    size_t num_to_read = std::min(static_cast<size_t>(n), max_readable);

    std::vector<uint8_t> buffer(num_to_read * vector_dim);
    size_t offset_bytes = static_cast<size_t>(offset) * vector_dim * sizeof(uint8_t);
    file.seekg(8 + offset_bytes, std::ios::beg);

    file.read(reinterpret_cast<char*>(buffer.data()), buffer.size() * sizeof(uint8_t));
    if (file.gcount() != static_cast<std::streamsize>(buffer.size() * sizeof(uint8_t))) {
        std::cerr << "[Coordinator]: Failed to read the full data block\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
        return {};
    }

    std::vector<float> vectors;
    vectors.reserve(num_to_read * vector_dim);
    for (uint8_t val : buffer) {
        vectors.push_back(static_cast<float>(val));
    }

    return vectors;
}

std::vector<float> readFbin(const std::string& filename, size_t vector_dim, int n, int offset) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[Coordinator]: Unable to open fbin file\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
        return {};
    }

    uint32_t num_points, dim_in_file;
    file.read(reinterpret_cast<char*>(&num_points), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&dim_in_file), sizeof(uint32_t));

    if (dim_in_file != vector_dim) {
        std::cerr << "[Coordinator]: Dimension mismatch: file " << filename << " says " << dim_in_file
                  << ", expected " << vector_dim << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
        return {};
    }

    if (offset < 0 || static_cast<size_t>(offset) >= num_points) {
        std::cerr << "[Coordinator]: Invalid offset\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
        return {};
    }

    size_t max_readable = num_points - static_cast<size_t>(offset);
    size_t num_to_read = std::min(static_cast<size_t>(n), max_readable);

    std::vector<float> buffer(num_to_read * vector_dim);
    size_t offset_bytes = static_cast<size_t>(offset) * vector_dim * sizeof(float);
    file.seekg(8 + offset_bytes, std::ios::beg);

    file.read(reinterpret_cast<char*>(buffer.data()), buffer.size() * sizeof(float));
    if (file.gcount() != static_cast<std::streamsize>(buffer.size() * sizeof(float))) {
        std::cerr << "[Coordinator]: Failed to read full data block from fbin\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
        return {};
    }

    return buffer;
}

std::vector<std::vector<int>> readGTIvecs(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[Coordinator]: Unable to open Ivecs file\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
        return std::vector<std::vector<int>>();
    }

    std::vector<std::vector<int>> vectors;

    while (!file.eof()) {
        int dim;
        file.read(reinterpret_cast<char*>(&dim), 4);
        if (file.eof()) break;

        std::vector<int> vec(dim);
        file.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(int));

        if (!file) {
            std::cerr << "[Coordinator]: Error reading ground truth data from file.\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
            return {};
        }

        vectors.push_back(std::move(vec));
    }

    return vectors;
}

std::vector<std::vector<int>> readGTBin(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[Coordinator]: Unable to open ground truth .bin file\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
        return {};
    }

    uint32_t num_queries, K;
    file.read(reinterpret_cast<char*>(&num_queries), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&K), sizeof(uint32_t));

    size_t total_elements = static_cast<size_t>(num_queries) * K;

    std::vector<uint32_t> all_ids(total_elements);
    file.read(reinterpret_cast<char*>(all_ids.data()), total_elements * sizeof(uint32_t));
    if (!file) {
        std::cerr << "[Coordinator]: Failed to read neighbor IDs\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
        return {};
    }

    file.seekg(total_elements * sizeof(float), std::ios::cur);

    std::vector<std::vector<int>> gt;
    gt.reserve(num_queries);
    for (uint32_t i = 0; i < num_queries; ++i) {
        std::vector<int> vec(K);
        for (uint32_t j = 0; j < K; ++j) {
            vec[j] = static_cast<int>(all_ids[i * K + j]);
        }
        gt.push_back(std::move(vec));
    }

    return gt;
}

std::vector<float> readVecs(const std::string& filename, size_t vector_dim, int n, int offset) {
    FileFormat format = getFileFormat(filename);
    std::vector<float> result;
    switch (format) {
        case BVECS:
            result = readBvecs(filename, vector_dim, n, offset);
            break;
        case FVECS:
            result = readFvecs(filename, vector_dim, n, offset);
            break;
        case I8BIN:
            result = readI8bin(filename, vector_dim, n, offset);
            break;
        case U8BIN:
            result = readU8bin(filename, vector_dim, n, offset);
            break;
        case FBIN:
            result = readFbin(filename, vector_dim, n, offset);
            break;
        default:
            std::cerr << "Unsupported file format " << filename << "\n";
            exit(1);
    }
    return result;
}

std::vector<std::vector<int>> readGT(const std::string& filename, FileFormat format) {
    std::vector<std::vector<int>> gt;
    switch (format) {
        case BVECS:
        case FVECS:
            gt = readGTIvecs(filename);
            break;
        case U8BIN:
        case I8BIN:
        case FBIN:
            gt = readGTBin(filename);
            break;
        default:
            std::cerr << "Unsupported file format " << filename << "\n";
            exit(1);
    }
    return gt;
}

int maximum_matching(const std::vector<std::vector<int>>& cost, std::vector<int>& assignment) {
    int n = cost.size();
    std::vector<int> u(n + 1), v(n + 1), p(n + 1), way(n + 1);
    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        std::vector<int> minv(n + 1, std::numeric_limits<int>::max());
        std::vector<bool> used(n + 1, false);
        int j0 = 0;
        do {
            used[j0] = true;
            int i0 = p[j0], delta = std::numeric_limits<int>::max(), j1 = -1;
            for (int j = 1; j <= n; ++j) {
                if (!used[j]) {
                    int cur = -cost[i0 - 1][j - 1] - u[i0] - v[j];
                    if (cur < minv[j]) {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if (minv[j] < delta) {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for (int j = 0; j <= n; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);
        do {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }

    assignment.resize(n);
    for (int j = 1; j <= n; ++j)
        if (p[j] != 0)
            assignment[p[j] - 1] = j - 1;

    return v[0];
}

int kmeans(
    float* sample,
    size_t n_points,
    size_t dim,
    size_t n_centers,
    float* centers,
    int* counts,
    float EPSILON,
    int EPOCHS,
    bool verbose
) {
    if (sample == nullptr || centers == nullptr || counts == nullptr || n_points == 0 || n_centers == 0 || dim == 0 || EPOCHS <= 0) {
        return -1;
    }

    static thread_local std::mt19937 gen(std::random_device{}());

    int num_iterations = -1;

    std::vector<float> distances(n_points, std::numeric_limits<float>::max());
    std::uniform_int_distribution<size_t> distrib(0, n_points - 1);

    size_t first = distrib(gen);
    std::copy(sample + first * dim, sample + (first + 1) * dim, centers);

    for (size_t c = 1; c < n_centers; ++c) {
        float total_weight = 0.0f;

        for (size_t i = 0; i < n_points; ++i) {
            float dist = computeEuclideanDistance(sample + i * dim, centers + (c - 1) * dim, dim);
            if (dist < distances[i]) {
                distances[i] = dist;
            }
            total_weight += distances[i];
        }

        size_t next_center = distrib(gen);
        if (total_weight > 0.0f) {
            float r = std::uniform_real_distribution<float>(0.0f, total_weight)(gen);
            float prefix_sum = 0.0f;
            for (size_t i = 0; i < n_points; ++i) {
                prefix_sum += distances[i];
                if (prefix_sum >= r) {
                    next_center = i;
                    break;
                }
            }
        }

        std::copy(sample + next_center * dim, sample + (next_center + 1) * dim, centers + c * dim);
    }

    std::vector<float> new_centers(n_centers * dim);
    std::vector<int> global_counts(n_centers);

    for (int iter = 0; iter < EPOCHS; ++iter) {
        if (verbose) {
            std::cout << "kmeans: " << iter << "/" << EPOCHS << "\n";
        }

        std::fill(new_centers.begin(), new_centers.end(), 0.0f);
        std::fill(global_counts.begin(), global_counts.end(), 0);

#pragma omp parallel
        {
            std::vector<float> local_centers(n_centers * dim, 0.0f);
            std::vector<int> thread_counts(n_centers, 0);

#pragma omp for nowait schedule(static)
            for (size_t i = 0; i < n_points; i++) {
                float* point = sample + i * dim;

                size_t best_center = 0;
                float best_dist = computeEuclideanDistance(point, centers, dim);

                for (size_t c = 1; c < n_centers; c++) {
                    float dist = computeEuclideanDistance(point, centers + c * dim, dim);
                    if (dist < best_dist) {
                        best_dist = dist;
                        best_center = c;
                    }
                }

                float* dst_center = local_centers.data() + best_center * dim;
                for (size_t d = 0; d < dim; d++) {
                    dst_center[d] += point[d];
                }
                thread_counts[best_center]++;
            }

#pragma omp critical
            {
                for (size_t c = 0; c < n_centers; c++) {
                    for (size_t d = 0; d < dim; d++)
                        new_centers[c * dim + d] += local_centers[c * dim + d];
                    global_counts[c] += thread_counts[c];
                }
            }
        }

        bool converged = true;

#pragma omp parallel for schedule(static) reduction(&& : converged)
        for (size_t c = 0; c < n_centers; c++) {
            float* center = centers + c * dim;
            float* next_center = new_centers.data() + c * dim;

            if (global_counts[c] > 0) {
                for (size_t d = 0; d < dim; d++) {
                    next_center[d] /= global_counts[c];
                }

                float shift = 0.0f;
                for (size_t d = 0; d < dim; d++) {
                    float delta = center[d] - next_center[d];
                    shift += delta * delta;
                }
                if (shift > EPSILON) {
                    converged = false;
                }

                std::copy(next_center, next_center + dim, center);
            }
        }

        if (converged) {
            num_iterations = iter;
            break;
        }

        if (iter == EPOCHS - 1) {
            num_iterations = iter;
        }
    }

    std::copy(global_counts.begin(), global_counts.end(), counts);

    return num_iterations;
}

std::vector<RunbookStep> load_runbook(const std::string& path,
                                      const std::string& dataset_key) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open runbook: " + path);

    std::map<int, RunbookStep> step_map;
    bool        in_dataset = false;
    int         cur_step   = -1;
    RunbookStep cur;

    std::string line;
    while (std::getline(f, line)) {
        if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;

        int indent = 0;
        while (indent < (int)line.size() && line[indent] == ' ') ++indent;
        const std::string content = line.substr(indent);

        if (indent == 0) {
            // Top-level key line (e.g. "msturing-100M-clustered:")
            if (cur_step >= 0) { step_map[cur_step] = cur; cur_step = -1; }
            size_t colon = content.find(':');
            std::string key = runbook_detail::trim(
                colon != std::string::npos ? content.substr(0, colon) : content);
            in_dataset = (key == dataset_key);

        } else if (in_dataset && indent == 2) {
            // Step number ("1:") or metadata ("max_pts:")
            if (cur_step >= 0) { step_map[cur_step] = cur; cur_step = -1; }
            size_t colon = content.find(':');
            if (colon == std::string::npos) continue;
            std::string key = runbook_detail::trim(content.substr(0, colon));
            bool all_digits = !key.empty() &&
                std::all_of(key.begin(), key.end(),
                            [](unsigned char c){ return std::isdigit(c); });
            if (all_digits) { cur_step = std::stoi(key); cur = RunbookStep{}; cur.step_num = cur_step; }

        } else if (in_dataset && indent == 4 && cur_step >= 0) {
            // Step field ("operation:", "start:", "end:")
            size_t colon = content.find(':');
            if (colon == std::string::npos) continue;
            std::string key = runbook_detail::trim(content.substr(0, colon));
            std::string val = runbook_detail::trim(content.substr(colon + 1));
            if      (key == "operation") cur.operation = val;
            else if (key == "start")     cur.start     = std::stoi(val);
            else if (key == "end")       cur.end       = std::stoi(val);
        }
    }
    if (cur_step >= 0) step_map[cur_step] = cur;

    std::vector<RunbookStep> result;
    result.reserve(step_map.size());
    for (auto& [k, v] : step_map) result.push_back(v);
    return result;
}