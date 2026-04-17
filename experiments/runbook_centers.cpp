#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <filesystem>
#include <yaml-cpp/yaml.h>
#include <omp.h>
#include "utils.h"
#include "hnswlib.h"

constexpr double MIN_EDGE_DIST = 1e-4; 

int main(int argc, char **argv) {
    // arguments: dataset num_centers output_dir
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <dataset> <num_centers> <output_dir>\n";
        return 1;
    }

    std::string dataset_name = argv[1];
    int num_centers = std::stoi(argv[2]);
    std::string output_dir = argv[3];

    std::error_code fs_ec;
    std::filesystem::create_directories(output_dir, fs_ec);
    if (fs_ec) {
        std::cerr << "Failed to create output directory '" << output_dir
                  << "': " << fs_ec.message() << "\n";
        return 1;
    }

    if (DATASETS.find(dataset_name) == DATASETS.end()) {
        std::cerr << "Unknown dataset: " << dataset_name << "\n";
        return 1;
    }

    std::string base_file = DATASETS[dataset_name]["base_file"];
    std::pair<int,int> data_info = get_dataset_info(base_file);
    int nvectors = data_info.first;
    int dim = data_info.second;

    std::string runbook_file = DATASETS[dataset_name]["runbook"];

    // read in runbook steps
    YAML::Node root = YAML::LoadFile(runbook_file);
    // Access top-level key
    YAML::Node steps = root[dataset_name];
    if (!steps) {
        std::cerr << "Key '" << dataset_name << "' not found!\n";
        return 1;
    }

    std::vector<int> vecid_to_centerid(nvectors, -1);
    std::vector<float> centers;          // flat: centers[c * dim + d]
    std::vector<double> cluster_sums;    // flat: cluster_sums[c * dim + d], float64 accumulator
    std::vector<int> center_counts;

    hnswlib::L2Space space(dim);

    hnswlib::HierarchicalNSW<float>* curr_hnsw = nullptr;

    // Hoist thread accumulator buffers outside the step loop to avoid
    // repeated allocation/deallocation of potentially large arrays.
    int num_threads = omp_get_max_threads();
    std::vector<double> thread_center_sums(num_threads * num_centers * dim, 0.0);
    std::vector<int>    thread_center_counts(num_threads * num_centers, 0);

    for (const auto& pair : steps) {
        std::string key = pair.first.as<std::string>();
        int stepNum;
        try {
            stepNum = std::stoi(key);
        } catch (const std::exception&) {
            std::cerr << "Skipping non-numeric runbook key: '" << key << "'\n";
            continue;
        }

        const YAML::Node& step = pair.second;
        if (!step.IsMap() || step.size() == 0) {
            std::cout << "Invalid Step: " << stepNum << "\n";
            continue;
        }

        std::cout << "Step " << stepNum << ":\n";

        std::string op = step["operation"].as<std::string>();
        if (op == "insert") {
            int start = step["start"].as<int>();
            int end   = step["end"].as<int>();
            std::cout << "  Insert Start: " << start << ", End: " << end << "\n";

            std::vector<float> new_vecs = readVecs(base_file, dim, end - start, start);

            if (centers.empty()) {
                // initialize centers with first batch
                centers.resize(num_centers * dim, 0.0f);
                center_counts.resize(num_centers, 0);
                int sample_size = std::min((size_t)(end-start), (size_t)100000);
                // get random sample of 0-based indices into new_vecs
                std::vector<size_t> indices(end-start);
                std::iota(indices.begin(), indices.end(), 0);
                std::shuffle(indices.begin(), indices.end(), std::mt19937(std::random_device{}()));
                indices.resize(sample_size);
                std::vector<float> sample(sample_size * dim);
                for (size_t i = 0; i < sample_size; i++) {
                    std::copy(new_vecs.begin() + indices[i] * dim, new_vecs.begin() + (indices[i] + 1) * dim, sample.begin() + i * dim);
                }
                // run kmeans on sample to get initial centers
                int num_iterations = kmeans(sample.data(), sample_size, dim, num_centers, centers.data(), center_counts.data());

                // build hnsw on centers
                curr_hnsw = new hnswlib::HierarchicalNSW<float>(&space, num_centers);
                #pragma omp parallel for
                for (int i = 0; i < num_centers; i++) {
                    curr_hnsw->addPoint(centers.data() + i * dim, i);
                }

                center_counts.assign(num_centers, 0); // reset counts; will be updated in insert loop below
                cluster_sums.assign(num_centers * dim, 0.0); // reset sums; will be accumulated in insert loop below
            }

            // Reset shared accumulator buffers (reused across steps).
            std::fill(thread_center_sums.begin(), thread_center_sums.end(), 0.0);
            std::fill(thread_center_counts.begin(), thread_center_counts.end(), 0);

            // start timing here for insert step
            auto insert_start_time = std::chrono::high_resolution_clock::now();

            #pragma omp parallel for
            for (int i = 0; i < end - start; i++) {
                int tid = omp_get_thread_num();
                int nearest_center = curr_hnsw->searchKnn(new_vecs.data() + i * dim, 1).top().second;
                // Each index (start + i) is unique for this batch, so writes are independent.
                vecid_to_centerid[start + i] = nearest_center;

                double* ts = thread_center_sums.data() + (tid * num_centers + nearest_center) * dim;
                for (int d = 0; d < dim; d++) {
                    ts[d] += new_vecs[i * dim + d];
                }
                thread_center_counts[tid * num_centers + nearest_center]++;
            }
            auto insert_end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> insert_duration = insert_end_time - insert_start_time;
            std::cout << "  Insert time: " << insert_duration.count() << " seconds\n";

            // Reduce per-thread accumulators, accumulate into cluster_sums,
            // recompute centers = cluster_sums / counts, and rebuild hnsw.
            auto rebuild_start_time = std::chrono::high_resolution_clock::now();
            delete curr_hnsw;
            curr_hnsw = new hnswlib::HierarchicalNSW<float>(&space, num_centers);
            #pragma omp parallel for
            for (int c = 0; c < num_centers; c++) {
                int new_count = 0;
                for (int t = 0; t < num_threads; t++) {
                    new_count += thread_center_counts[t * num_centers + c];
                }
                int total = center_counts[c] + new_count;
                double* cs = cluster_sums.data() + c * dim;
                for (int d = 0; d < dim; d++) {
                    double sum = 0.0;
                    for (int t = 0; t < num_threads; t++) {
                        sum += thread_center_sums[(t * num_centers + c) * dim + d];
                    }
                    cs[d] += sum;
                }
                int divisor = std::max(total, 1);
                float* center = centers.data() + c * dim;
                for (int d = 0; d < dim; d++) {
                    center[d] = static_cast<float>(cs[d] / divisor);
                }
                center_counts[c] = total;
            }

            #pragma omp parallel for
            for (int i = 0; i < num_centers; i++) {
                curr_hnsw->addPoint(centers.data() + i * dim, i);
            }

            auto rebuild_end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> rebuild_duration = rebuild_end_time - rebuild_start_time;
            std::cout << "  HNSW rebuild time: " << rebuild_duration.count() << " seconds\n";
            std::chrono::duration<double> total_insert_duration = rebuild_end_time - insert_start_time;
            std::cout << "  Total insert step time: " << total_insert_duration.count() << " seconds\n";

        } else if (op == "delete") {
            int start = step["start"].as<int>();
            int end   = step["end"].as<int>();
            std::cout << "  Delete Start: " << start << ", End: " << end << "\n";
            if (curr_hnsw == nullptr || centers.empty()) {
                std::cerr << "Warning: delete at step " << stepNum << " before any insert; skipping.\n";
                continue;
            }
            std::vector<float> deleted_vecs = readVecs(base_file, dim, end - start, start);
            int num_to_delete = end - start;

            // Reset shared accumulator buffers (reused across steps).
            std::fill(thread_center_sums.begin(), thread_center_sums.end(), 0.0);
            std::fill(thread_center_counts.begin(), thread_center_counts.end(), 0);

            auto delete_start_time = std::chrono::high_resolution_clock::now();
            #pragma omp parallel for
            for (int i = 0; i < num_to_delete; i++) {
                int tid = omp_get_thread_num();
                int vec_id = start + i;

                if (vec_id < 0 || vec_id >= nvectors) {
                    continue;
                }

                int center_id = vecid_to_centerid[vec_id];
                if (center_id < 0 || center_id >= num_centers) {
                    continue;
                }

                thread_center_counts[tid * num_centers + center_id]++;
                double* ts = thread_center_sums.data() + (tid * num_centers + center_id) * dim;
                for (int d = 0; d < dim; d++) {
                    ts[d] += deleted_vecs[i * dim + d];
                }

                // Tombstone immediately; each vec_id in [start, end) is unique in this step.
                vecid_to_centerid[vec_id] = -1;
            }

            auto delete_end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> delete_duration = delete_end_time - delete_start_time;
            std::cout << "  Delete time: " << delete_duration.count() << " seconds\n";

            // Reduce per-thread accumulators, subtract from cluster_sums,
            // recompute centers = cluster_sums / counts, and rebuild hnsw.
            auto rebuild_start_time = std::chrono::high_resolution_clock::now();
            delete curr_hnsw;
            curr_hnsw = new hnswlib::HierarchicalNSW<float>(&space, num_centers);
            #pragma omp parallel for
            for (int c = 0; c < num_centers; c++) {
                int del_count = 0;
                for (int t = 0; t < num_threads; t++) {
                    del_count += thread_center_counts[t * num_centers + c];
                }

                if (del_count > 0) {
                    int old_count = center_counts[c];
                    if (del_count > old_count) {
                        del_count = old_count;
                    }
                    int remaining = old_count - del_count;

                    double* cs = cluster_sums.data() + c * dim;
                    for (int d = 0; d < dim; d++) {
                        double del_sum = 0.0;
                        for (int t = 0; t < num_threads; t++) {
                            del_sum += thread_center_sums[(t * num_centers + c) * dim + d];
                        }
                        cs[d] -= del_sum;
                    }

                    int divisor = std::max(remaining, 1);
                    float* center = centers.data() + c * dim;
                    for (int d = 0; d < dim; d++) {
                        center[d] = static_cast<float>(cs[d] / divisor);
                    }
                    center_counts[c] = remaining;
                }
            }

            #pragma omp parallel for
            for (int i = 0; i < num_centers; i++) {
                curr_hnsw->addPoint(centers.data() + i * dim, i);
            }

            auto rebuild_end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> rebuild_duration = rebuild_end_time - rebuild_start_time;
            std::cout << "  HNSW rebuild time: " << rebuild_duration.count() << " seconds\n";
            std::chrono::duration<double> total_delete_duration = rebuild_end_time - delete_start_time;
            std::cout << "  Total delete step time: " << total_delete_duration.count() << " seconds\n";
        } else {
            if (curr_hnsw == nullptr || centers.empty()) {
                std::cerr << "Warning: query/save at step " << stepNum << " before any insert; skipping.\n";
                continue;
            }
            // save the current centers, center counts, hnsw state, vec_to_center mapping, etc. for this step
            // pad step num with zeros so they are sorted with 6 figs
            std::ostringstream step_stream;
            step_stream << std::setw(6) << std::setfill('0') << stepNum;
            std::string step_str = step_stream.str();

            std::string center_file = output_dir + "/step_" + step_str + "_centers.csv";
            std::string count_file = output_dir + "/step_" + step_str + "_center_counts.csv";
            std::string mapping_file = output_dir + "/step_" + step_str + "_vecid_to_centerid.bin";
            std::string hnsw_file = output_dir + "/step_" + step_str + "_hnsw.bin";
            // after saving the mapping file, delete the previous one
            std::ofstream center_out(center_file, std::ios::out);
            std::ofstream count_out(count_file, std::ios::out);
            std::ofstream mapping_out(mapping_file, std::ios::out | std::ios::binary);

            for (int c = 0; c < num_centers; c++) {
                const float* center = centers.data() + c * dim;
                for (int d = 0; d < dim; d++) {
                    center_out << center[d];
                    if (d < dim - 1) {
                        center_out << ",";
                    }
                }
                center_out << "\n";
                count_out << center_counts[c] << "\n";
            }

            mapping_out.write(reinterpret_cast<char*>(vecid_to_centerid.data()), vecid_to_centerid.size() * sizeof(int));
            curr_hnsw->saveIndex(hnsw_file);

            // TODO: extract base layer of hnsw and save
            // Save base layer (level 0) neighbors
            std::string base_layer_file = output_dir + "/step_" + step_str + "_base_layer.csv";
            std::ofstream base_layer_out(base_layer_file);
            
            for (size_t internal_id = 0; internal_id < (size_t)num_centers; internal_id++) {
                auto label_i = curr_hnsw->getExternalLabel(internal_id);
 
                // Low-level access to base layer
                auto bottom = curr_hnsw->get_linklist_at_level(internal_id, 0);
                int nlinks = curr_hnsw->getListCount(bottom);
                hnswlib::tableint *links = (hnswlib::tableint *)(bottom + 1);
 
                const float* vec_i = centers.data() + label_i * dim;
 
                // output edge weight == distance between i, j
                for (int j = 0; j < nlinks; j++) {
                    auto label_j = curr_hnsw->getExternalLabel(links[j]);
                    const float* vec_j = centers.data() + label_j * dim;
                    float dist = computeEuclideanDistance(vec_i, vec_j, dim);
 
                    if (!std::isfinite(dist)) continue;
                    if (dist < MIN_EDGE_DIST) continue;
 
                    base_layer_out << label_i << "," << label_j << "," << dist << "\n";
                }
            }

            base_layer_out.close();

            // remove previous mapping file if it exists
            // TODO
            // if (std::filesystem::exists(prev_mapping_file)) {
            //     std::filesystem::remove(prev_mapping_file);
            // }
        }
    }
}