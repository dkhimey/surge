#include "index.h"
#include <falconn/lsh_nn_table.h>

#include <iostream>
#include <functional>
#include <algorithm>
#include <limits>
#include <set>
#include <unordered_set>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <omp.h>

typedef falconn::DenseVector<float> Point;
Point transformVector(float* vec, size_t dim, float alpha = 0.5);

#define TAG_SEND_NUM 100000001
#define TAG_SEND_VECS 100000002
#define TAG_SEND_LABELS 100000003

// ========== Coordinator Implementation ==========
Coordinator::Coordinator(
        int dim,
        Log* logger)
        : dim_(dim),
    comm_(nullptr),
        logger_(logger),
        gen_(std::random_device{}())
{
    
    space_ = new hnswlib::L2Space(dim_);
    computeDistance_ = [this](float* a, float* b) { return computeEuclideanDistance(a, b, dim_); };

    std::cout << "[Coordinator]: Instantiated.\n";
}


Coordinator::Coordinator(
        int dim,
        Communicator* comm,
        Log* logger)
        : dim_(dim),
        comm_(comm),
        logger_(logger),
        gen_(std::random_device{}())
{
    
    space_ = new hnswlib::L2Space(dim_);
    computeDistance_ = [this](float* a, float* b) { return computeEuclideanDistance(a, b, dim_); };

    std::cout << "[Coordinator]: Instantiated.\n";
}

void Coordinator::setSampleData(float* data, size_t count) {
    sample_data_ = data;
    sample_count_ = count;

    std::cout << "[Coordinator] Sample data set with " << count << " elements\n";
}

std::pair<std::vector<int>, std::vector<int>> Coordinator::getBottomLayer_(hnswlib::HierarchicalNSW<float>* graph) {
    if (graph == nullptr) graph = meta_HNSW_;

    std::unordered_map<int, std::unordered_set<int>> adj_map;


    {
        if (graph == meta_HNSW_) std::shared_lock lock(graph_mutex_);
        for (int i = 0; i < ncenters_; i++) {
            auto bottom = graph->get_linklist_at_level(i, 0);
            int nlinks = graph->getListCount(bottom);
            hnswlib::tableint *links = (hnswlib::tableint *)(bottom + 1);

            for (int j = 0; j < nlinks; j++) {
                int neighbor = links[j];
                if (i == neighbor) continue; // Skip self-loops

                // Insert both directions, automatically skipping duplicates
                adj_map[i].insert(neighbor);
                adj_map[neighbor].insert(i);
            }
        }
    }

    // Step 2: Convert to CSR format
    std::vector<int> xadj;
    std::vector<int> adjncy;
    int pos = 0;

    xadj.push_back(0); // xadj[0] = 0
    for (int i = 0; i < ncenters_; i++) {
        const auto &neighbors = adj_map[i];
        for (int neighbor : neighbors) {
            adjncy.push_back(neighbor);
            pos++;
        }
        xadj.push_back(pos); // xadj[i+1]
    }


    return {xadj, adjncy};
}

void Coordinator::setEfSearch(int ef_search) {
    std::unique_lock lock(graph_mutex_);
    meta_HNSW_->setEf(ef_search);
}

void Coordinator::build(
        int ncenters,
        int num_partitions,
        int ef_construction,
        int M_meta  
    ) {

    this->num_partitions_ = num_partitions;
    this->ncenters_ = ncenters;

    if (sample_count_ == 0) {
        std::cerr << "[Coordinator]: set sample data before building\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // 2. run k-means with m centers
    double start = MPI_Wtime();
    int num_iterations;

    centers_ = std::vector<float>(ncenters * dim_);
    num_iterations = kmeans_(sample_data_, sample_count_, ncenters, centers_.data());
    double end = MPI_Wtime();

    logger_->kmeans_time = end - start;
    logger_->kmeans_num_iterations = num_iterations;

    // 3. build meta-HNSW on the centers
    start = MPI_Wtime();
    meta_HNSW_ = new hnswlib::HierarchicalNSW<float>(space_, ncenters, M_meta, ef_construction);
    meta_HNSW_->addPoint(centers_.data(), 0); // first element adding not thread safe.
    // #pragma omp parallel for
    for (int i = 1; i < ncenters; i++) {
        meta_HNSW_->addPoint(centers_.data() + i * dim_, i);
    }
    end = MPI_Wtime();

    logger_->index_build_time = end - start;

    meta_HNSW_->setEf(ef_construction);

    logger_->logCenters(centers_);
    // delete[] centers;

    // 4. partition the bottom layer into w partitions
    // Step 1: Build symmetric adjacency list without duplicates
    start = MPI_Wtime();
    std::pair<std::vector<int>, std::vector<int>> ret = getBottomLayer_();
    std::vector<int> xadj = ret.first;
    std::vector<int> adjncy = ret.second;

    // Step 3: Run partitioning
    int edge_cut = 0; // TODO
    int ncenters_int = (int) ncenters;
    int w_partitions_int = (int) num_partitions;
    double imbalance = 0.03; // TODO
    partitions = std::vector<int>(ncenters, -1);
    int seed = 0;

    std::cout << "NUMBER OF PARTITIONS: " << num_partitions << std::endl;
    // run Karlsuhe partitioning algorithm (no vertex weights, matching rePartition)
    kaffpa(&ncenters_int, nullptr, xadj.data(), nullptr, adjncy.data(),
           &w_partitions_int, &imbalance, true, seed, STRONG, &edge_cut, partitions.data());

    end = MPI_Wtime();
    logger_->karlsuhe_time = end - start;

    logger_->logPartitions(partitions);
    std::cout << "EDGE CUT: " << edge_cut << "\n";
    size_t total_edges = adjncy.size() / 2;
    std::cout << "EDGE CUT RATIO: " << edge_cut / float(total_edges) << "\n";
    logger_->edge_cut_ratio = edge_cut / double(total_edges);
}

std::vector<std::pair<float, hnswlib::labeltype>> Coordinator::findClosestCenters_(float* vec, size_t n) {
    std::shared_lock lock(graph_mutex_); // protect meta_HNSW access
    std::vector<std::pair<float, hnswlib::labeltype>> centers;
    if (meta_HNSW_ == nullptr) {
        std::cerr << "[Coordinator] HNSW not initialized.\n";
    }

    centers = meta_HNSW_->searchKnnCloserFirst(vec, n);
    return centers;
}

std::vector<size_t> Coordinator::convertCentersToPartitions_(const std::vector<std::pair<float, hnswlib::labeltype>>& centers, float* dist) {
    // Return partitions in nearest-centroid order (centers is already sorted
    // closer-first by searchKnnCloserFirst), deduplicating via a seen set.
    std::vector<size_t> ret_partitions;
    std::unordered_set<int> seen;
    for (const auto& ref : centers) {
        int pid = partitions[ref.second];
        if (seen.insert(pid).second) {
            ret_partitions.push_back(static_cast<size_t>(pid));
        }
    }

    if (dist) *dist = centers[0].first;

    return ret_partitions;
}

void Coordinator::updateCentersForInsert_(float* vec, size_t label, size_t closest_center_label) {
    int closest_center_idx = closest_center_label * dim_;

    {
        std::unique_lock lock(center_mutex_);
        label_to_center_[label] = closest_center_label;
        for (int i = 0; i < dim_; i++) {
            float orig_center_i = centers_[closest_center_idx + i];
            int count = center_counts_[closest_center_label];

            // float new_center_i = (centers[closest_center_idx + i] * center_counts_[closest_center_label] + vec[i]) / (center_counts_[closest_center_label] + 1);
            centers_[closest_center_idx + i] = (orig_center_i * count + vec[i]) / (count + 1);
        }
        center_counts_[closest_center_label]++;
    }
}

void Coordinator::updateCentersForInsertBatch(const std::vector<float>& vecs,
                                               const std::vector<int>& center_ids) {
    // Phase 1 – parallel accumulation.  Each OMP thread builds a thread-local
    // map of (center_id → {sum_vector, count}).  No lock is held here.
    const int n = static_cast<int>(center_ids.size());
    using Accum = std::unordered_map<int, std::pair<std::vector<float>, int>>;
    Accum global;

    #pragma omp parallel
    {
        Accum local;

        #pragma omp for nowait schedule(static)
        for (int i = 0; i < n; i++) {
            int cid = center_ids[i];
            auto& [sum, cnt] = local[cid];
            if (sum.empty()) sum.assign(dim_, 0.0f);
            const float* v = vecs.data() + static_cast<size_t>(i) * dim_;
            for (size_t d = 0; d < dim_; d++) sum[d] += v[d];
            cnt++;
        }

        // Merge thread-local map into global under a critical section.
        #pragma omp critical
        for (auto& [cid, p] : local) {
            auto& [lsum, lcnt] = p;
            auto& [gsum, gcnt] = global[cid];
            if (gsum.empty()) gsum.assign(dim_, 0.0f);
            for (size_t d = 0; d < dim_; d++) gsum[d] += lsum[d];
            gcnt += lcnt;
        }
    }

    // Phase 2 – single write lock, one pass over only the touched centers.
    std::unique_lock<std::shared_mutex> lk(center_mutex_);
    for (auto& [cid, p] : global) {
        auto& [sum, new_k] = p;
        const int old_n  = center_counts_[cid];
        const int total_n = old_n + new_k;
        float* c = centers_.data() + static_cast<size_t>(cid) * dim_;
        for (size_t d = 0; d < dim_; d++)
            c[d] = (old_n * c[d] + sum[d]) / static_cast<float>(total_n);
        center_counts_[cid] = total_n;
    }
}

void Coordinator::updateCentersForDelete_(std::vector<float>& vec, int closest_center_label) {
    int closest_center_idx = closest_center_label * dim_;

    {
        std::unique_lock lock(center_mutex_);
        int count = center_counts_[closest_center_label];
        for (int i = 0; i < dim_; i++) {
            float orig_center_i = centers_[closest_center_idx + i];

            centers_[closest_center_idx + i] = (orig_center_i * count - vec[i]) / (count - 1);
        }
        center_counts_[closest_center_label]--;
    }
}

void Coordinator::updateCentersForDeleteBatch(const std::vector<float>& vecs,
                                               const std::vector<int>& closest_center_labels) {
    // Phase 1 – parallel accumulation of deleted-vector sums per center.
    const int n = static_cast<int>(closest_center_labels.size());
    using Accum = std::unordered_map<int, std::pair<std::vector<float>, int>>;
    Accum global;

    #pragma omp parallel
    {
        Accum local;

        #pragma omp for nowait schedule(static)
        for (int i = 0; i < n; i++) {
            int cid = closest_center_labels[i];
            auto& [sum, cnt] = local[cid];
            if (sum.empty()) sum.assign(dim_, 0.0f);
            const float* v = vecs.data() + static_cast<size_t>(i) * dim_;
            for (size_t d = 0; d < dim_; d++) sum[d] += v[d];
            cnt++;
        }

        #pragma omp critical
        for (auto& [cid, p] : local) {
            auto& [lsum, lcnt] = p;
            auto& [gsum, gcnt] = global[cid];
            if (gsum.empty()) gsum.assign(dim_, 0.0f);
            for (size_t d = 0; d < dim_; d++) gsum[d] += lsum[d];
            gcnt += lcnt;
        }
    }

    // Phase 2 – single write lock, subtract accumulated sums.
    std::unique_lock<std::shared_mutex> lk(center_mutex_);
    for (auto& [cid, p] : global) {
        auto& [sum, del_k] = p;
        const int old_n = center_counts_[cid];
        const int new_n = old_n - del_k;
        float* c = centers_.data() + static_cast<size_t>(cid) * dim_;
        if (new_n > 0) {
            for (size_t d = 0; d < dim_; d++)
                c[d] = (old_n * c[d] - sum[d]) / static_cast<float>(new_n);
        }
        // If new_n == 0 the center is now empty; leave coordinates as-is.
        // The meta-HNSW still routes to it but it will receive no further
        // vectors until the next rebuild reassigns it.
        center_counts_[cid] = std::max(new_n, 0);
    }
}


std::vector<size_t> Coordinator::getPartitionsForSearch_Branching_(float* vec, int branching_factor, float* dist) {
    std::vector<std::pair<float, hnswlib::labeltype>> centers = findClosestCenters_(vec, branching_factor);
    return convertCentersToPartitions_(centers, dist);
}


std::vector<size_t> Coordinator::getPartitionsForSearch_Nprobe_(float* vec, int nprobe, float* dist) {
    // Mirrors visit_enforced_p: increase k until nprobe unique partitions are found,
    // then return them in the order they first appear among the nearest centers.
    size_t cur_k = static_cast<size_t>(std::min((size_t)(nprobe), (size_t)(ncenters_))); // start with a reasonably large k to reduce number of iterations; will be capped at ncenters_

    std::vector<std::pair<float, hnswlib::labeltype>> centers;
    while (true) {
        cur_k = std::min(cur_k, ncenters_);
        centers = findClosestCenters_(vec, cur_k);

        std::unordered_set<int> unique_partitions;
        for (const auto& c : centers) {
            unique_partitions.insert(partitions[c.second]);
        }

        if (static_cast<int>(unique_partitions.size()) >= nprobe || cur_k >= ncenters_) {
            break;
        }
        cur_k *= 10;
    }

    if (dist && !centers.empty()) {
        *dist = centers[0].first;
    }

    // Walk centers in nearest-first order; collect partitions until nprobe unique ones seen.
    std::vector<size_t> visited_partitions;
    std::unordered_set<int> seen;

    for (const auto& c : centers) {
        int pid = partitions[c.second];
        if (seen.insert(pid).second) {          // newly seen partition
            visited_partitions.push_back(static_cast<size_t>(pid));
            if (static_cast<int>(visited_partitions.size()) == nprobe) {
                break;
            }
        }
    }

    return visited_partitions;
}

std::vector<size_t> Coordinator::getPartitionsForSearch_RecallTgt_(float* vec, float recall_target, float* dist) {
    // search with a recall target
    size_t knn = std::min<size_t>(50, ncenters_);
    std::vector<std::pair<float, hnswlib::labeltype>> centers = findClosestCenters_(vec, knn);

    if (centers.empty()) {
        return {};
    }

    if (dist) {
        *dist = centers[0].first;
    }

    // Normalise all distances by the nearest-centroid distance so the
    // exponential weighting is scale-invariant across datasets.
    const double d0 = static_cast<double>(centers[0].first) + 1e-10;

    // Choose the size prior: if any center_counts_ entry is nonzero the
    // index has seen real vectors and we use per-center membership counts.
    // Otherwise (e.g. build() was called but no vectors were ever routed,
    // as in theoretical_partitioning_quality) fall back to the number of
    // centers assigned to each partition — always derivable from partitions[].
    bool counts_live = false;
    for (int c : center_counts_) { if (c > 0) { counts_live = true; break; } }

    std::vector<int> part_size;
    if (!counts_live) {
        part_size.assign(num_partitions_, 0);
        for (int p : partitions) ++part_size[static_cast<size_t>(p)];
    }

    std::vector<double> partition_probs(num_partitions_, 0.0);
    for (const auto& [d, center_id] : centers) {
        const int    pid     = partitions[center_id];
        const double rel_d   = static_cast<double>(d) / d0;
        const double size_wt = counts_live
            ? static_cast<double>(center_counts_[center_id])
            : static_cast<double>(part_size[static_cast<size_t>(pid)]);
        partition_probs[static_cast<size_t>(pid)] += size_wt * std::exp(-1.0 * rel_d);
    }

    const double prob_sum = std::accumulate(partition_probs.begin(), partition_probs.end(), 0.0);
    if (prob_sum <= 0.0) {
        return {static_cast<size_t>(partitions[centers[0].second])};
    }
    for (double& p : partition_probs) p /= prob_sum;

    std::vector<size_t> ordered_pids(num_partitions_);
    std::iota(ordered_pids.begin(), ordered_pids.end(), 0);
    std::sort(ordered_pids.begin(), ordered_pids.end(), [&](size_t a, size_t b) {
        return partition_probs[a] > partition_probs[b];
    });

    recall_target = std::clamp(recall_target, 0.0f, 1.0f);

    std::vector<size_t> visited_partitions;
    double recall_estimate = 0.0;
    for (size_t pid : ordered_pids) {
        if (partition_probs[pid] <= 0.0) break;
        visited_partitions.push_back(pid);
        recall_estimate += partition_probs[pid];
        if (recall_estimate >= recall_target) break;
    }

    if (visited_partitions.empty()) {
        visited_partitions.push_back(static_cast<size_t>(partitions[centers[0].second]));
    }

    return visited_partitions;
}

size_t Coordinator::getCurrentPartition(float* vec) {
    std::vector<std::pair<float, hnswlib::labeltype>> centers = findClosestCenters_(vec, 1);
    return convertCentersToPartitions_(centers, nullptr)[0];
}

std::vector<size_t> Coordinator::getPartitionsForInsert_(float* vec, size_t label, float* dist) {
    std::vector<std::pair<float, hnswlib::labeltype>> centers = findClosestCenters_(vec, 1);
    updateCentersForInsert_(vec, label, centers[0].second);
    return convertCentersToPartitions_(centers, dist);
}
std::vector<size_t> Coordinator::getPartitionsForDelete_(size_t label) {
    // std::vector<std::pair<float, hnswlib::labeltype>> centers = findClosestCenters_(vec, 1);
    int center_idx = label_to_center_[label];
    size_t partition = partitions[center_idx];
    return std::vector<size_t>{partition};
}

std::vector<int> Coordinator::distribute_vectors(
        const std::string& base_file, 
        int total_vectors, 
        bool log_partitions,
        int num_threads,
        std::vector<int>* preassigned_partitions
    ) {
    bool partitions_assigned = (preassigned_partitions != nullptr);
    if (num_threads == -1){
        printf("Using max threads: %d\n", omp_get_max_threads());
        num_threads = omp_get_max_threads();
    }

    omp_set_num_threads(num_threads);
    int vectors_per_thread = (total_vectors + num_threads - 1) / num_threads;

    std::vector<int> partition_assignments;
    std::vector<float> partition_distances;
    if (log_partitions) {
        partition_assignments = std::vector<int>(total_vectors, -1);
        partition_distances = std::vector<float>(total_vectors, -1);
    }

    std::vector<omp_lock_t> locks(num_partitions_ + 1);
    for (auto& l : locks) omp_init_lock(&l);

    std::cout << "num_threads = " << num_threads << ", omp_get_max_threads = " << omp_get_max_threads() << "\n";
    std::vector<int> counts_per_partition(num_partitions_, 0);
    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        // std::cout << "[Coordinator] Thread " << tid << " started for distributing vectors\n";
        int thread_start = tid * vectors_per_thread;
        int thread_end = std::min(thread_start + vectors_per_thread, total_vectors);


        // std::cout << "[Coordinator] Thread " << tid << " processing vectors " << thread_start << " to " << thread_end << "\n";
        for (int global_index = thread_start; global_index < thread_end; global_index += VECTOR_BATCH_SIZE) {
            if (tid == 0 && global_index % 10000 == 0)
                std::cout << "[Coordinator] - still sending " << MPI_Wtime() << "\n"; 
            int num_to_read = std::min(VECTOR_BATCH_SIZE, thread_end - global_index);
            std::vector<float> X = readVecs(base_file, dim_, num_to_read, global_index);

            std::vector<std::vector<float>> partition_vectors(num_partitions_ + 1);
            std::vector<std::vector<int>> partition_indices(num_partitions_ + 1);

            float center_dist = 0;
            for (size_t i = 0; i < num_to_read; ++i) {
                size_t vec_index = global_index + i;
                float* vec_ptr = X.data() + i * dim_;

                size_t p;
                if (partitions_assigned) {
                    p = (*preassigned_partitions)[vec_index];
                } else {
                    p = getPartitionsForInsert_(vec_ptr, vec_index, &center_dist)[0];
                }

                if (log_partitions) {
                    partition_assignments[vec_index] = p+1;
                    partition_distances[vec_index] = center_dist;
                }
                partition_vectors[p+1].insert(partition_vectors[p+1].end(), vec_ptr, vec_ptr + dim_);
                partition_indices[p+1].push_back(vec_index);
            }

            for (int p = 1; p < num_partitions_ + 1; ++p) {
                int vec_count = partition_indices[p].size();
                if (vec_count == 0) continue;

                MessageHeader header(VECTOR_SEND, vec_count);
                {
                    omp_set_lock(&locks[p]);
                    counts_per_partition[p-1] += vec_count;
                    MPI_Send(&header, sizeof(MessageHeader), MPI_BYTE, p, 0, MPI_COMM_WORLD);
                    MPI_Send(partition_vectors[p].data(), vec_count * dim_, MPI_FLOAT, p, 0, MPI_COMM_WORLD);
                    MPI_Send(partition_indices[p].data(), vec_count, MPI_INT, p, 0, MPI_COMM_WORLD);
                    omp_unset_lock(&locks[p]);
                }
            }
        }
    }

    if (log_partitions) {
        logger_->logPartitions(partition_assignments, true);
        logger_->logPartitionDists(partition_distances);
    }

    return counts_per_partition;
}

std::pair<int, std::vector<int>> Coordinator::matchPartitions_(const std::vector<int>& part1, const std::vector<int>& part2) {
    int n = part1.size();
    if (part2.size() != n) return {0, std::vector<int>()};

    // Build cost matrix: cost[i][j] = number of times part1[i] aligns with part2[j]
    std::vector<std::vector<int>> cost(num_partitions_, std::vector<int>(num_partitions_, 0));
    for (int i = 0; i < n; ++i)
        cost[part1[i]][part2[i]]++;

    // Use maximum matching algorithm to find best matching
    std::vector<int> assignment;
    int score = maximum_matching(cost, assignment);

    // int test = 0;
    // for (int i = 0; i < assignment.size(); i++) {
    //     for (int j = 0; j < assignment.size(); j++) {
    //         if (assignment[i] == j) continue;
    //         // std::cout << "      " << cost[i][j] << " vectors from " << j << " to " << assignment[i] << "\n";
    //         test+=cost[i][j];
    //     }
    // }
    std::cout << "TO MOVE: " << ncenters_ - score << " / " << ncenters_ << "\n";

    // Check if relabeled part1 matches part2
    // for (int i = 0; i < n; ++i) {
    //     if (assignment[part1[i]] != part2[i])
    //         return false;
    // }

    return {ncenters_ - score, assignment};
}

int Coordinator::rePartition(std::vector<int>& new_partitions, hnswlib::HierarchicalNSW<float>*& new_meta_HNSW, int ef_construction, int M_meta,
                              double* out_hnsw_s, double* out_bottom_s, double* out_kaffpa_s, double* out_relabel_s) {
    double start = MPI_Wtime();
    new_meta_HNSW = new hnswlib::HierarchicalNSW<float>(space_, ncenters_, M_meta, ef_construction);
    {
        std::unique_lock lock(center_mutex_);
        for (int i = 0; i < ncenters_; i++) {
            new_meta_HNSW->addPoint(centers_.data() + (i * dim_), i);
        }
    }
    new_meta_HNSW->setEf(200);
    double end = MPI_Wtime();
    double hnsw_time = end-start;

    start = MPI_Wtime();
    std::pair<std::vector<int>, std::vector<int>> ret = getBottomLayer_(new_meta_HNSW);
    std::vector<int> xadj = ret.first;
    std::vector<int> adjncy = ret.second;
    end = MPI_Wtime();
    double bottom_layer = end-start;

    int edge_cut = 0; // TODO
    int m_centers_int = (int) ncenters_;
    int w_partitions_int = (int) num_partitions_;
    double imbalance = 0.03; // TODO
    new_partitions = std::vector<int>(ncenters_, -1);
    int seed = gen_();
    // run partitioning algo
    start = MPI_Wtime();
    kaffpa(&m_centers_int, nullptr, xadj.data(), nullptr, adjncy.data(), 
           &w_partitions_int, &imbalance, true, seed, STRONG, &edge_cut, new_partitions.data());
    end = MPI_Wtime();
    double partition_time = end-start;

    start = MPI_Wtime();
    std::pair<int, std::vector<int>> matching = matchPartitions_(new_partitions, partitions);
    int to_move = matching.first;
    std::vector<int> relabel = matching.second;

    for (int i = 0; i < ncenters_; i++) {
        new_partitions[i] = relabel[new_partitions[i]];
    }
    end = MPI_Wtime();
    double partition_relabel = end-start;

    logger_->logRepartition(hnsw_time, bottom_layer, partition_time, partition_relabel);

    if (out_hnsw_s)    *out_hnsw_s    = hnsw_time;
    if (out_bottom_s)  *out_bottom_s  = bottom_layer;
    if (out_kaffpa_s)  *out_kaffpa_s  = partition_time;
    if (out_relabel_s) *out_relabel_s = partition_relabel;

    return to_move;
}

int Coordinator::reBuild(int world_size, int ef_construction, int M_meta, int full_threshold, int partial_threshold) {
    std::cout << "[Controller] rebuild? \n";
    RebuildState expected = IDLE;
    if (!rebuild_state.compare_exchange_strong(expected, REBUILDING)) { // todo: atomic check & swap
        rebuild_pending_ = true;
        return 0;
    }

    double start = MPI_Wtime();
    hnswlib::HierarchicalNSW<float>* new_meta_HNSW = nullptr;
    std::vector<int> new_partitions;
    int to_move = rePartition(new_partitions, new_meta_HNSW, ef_construction, M_meta);
    double end = MPI_Wtime();
    double total_repartition_time = end - start;

    if (to_move < full_threshold && to_move < partial_threshold) {

        RebuildState expected = REBUILDING;
        bool ok = rebuild_state.compare_exchange_strong(
            expected,
            IDLE
        );
        assert(ok);

        return 0;

        // rebuilding = false;
        // return 0;
    }

    std::cout << "[Controller] requesting rebuild\n";

    start = MPI_Wtime();
    assert(new_meta_HNSW != nullptr);
    new_meta_HNSW->saveIndex("tmp_hnsw.bin");

    std::ifstream infile("tmp_hnsw.bin", std::ios::binary);
    std::vector<char> buffer((std::istreambuf_iterator<char>(infile)),
                            std::istreambuf_iterator<char>());

    int meta_size = buffer.size();
    end = MPI_Wtime();
    double hnsw_serialization_time = end - start;

    start = MPI_Wtime();

    MessageType rebuild_type;
    if (to_move >= full_threshold) {
        rebuild_type = FULL_REBUILD_REQUEST;
    } else {
        rebuild_type = PARTIAL_REBUILD_REQUEST;
    }

    for (int i = 1; i < world_size; i++) {
        MessageHeader header; 
        header.type = rebuild_type;
        header.size = meta_size;
        MPI_Send(&header, sizeof(MessageHeader), MPI_BYTE, i, 0, MPI_COMM_WORLD);

        MPI_Send(buffer.data(), meta_size, MPI_BYTE, i, META_HNSW_SEND, MPI_COMM_WORLD);

        MPI_Send(new_partitions.data(), ncenters_, MPI_INT, i, META_PARTITIONS_SEND, MPI_COMM_WORLD);
    }

    for (int i = 1; i < world_size; i++) {
        MessageHeader header;
        MPI_Recv(&header, sizeof(MessageHeader), MPI_BYTE, i, REBUILD_SUCCESS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        assert(header.type == REBUILD_SUCCESS);
        // std::cout << "[Coordinator] : received REBUILD_SUCCESS from" << i << "\n";
    }
    end = MPI_Wtime();
    double reorganization_time = end - start;

    {
        std::unique_lock lock(graph_mutex_);
        free(meta_HNSW_);
        meta_HNSW_ = new_meta_HNSW;
        partitions = new_partitions;
    }

    logger_->logCoordinatorRebuild(rebuild_type, total_repartition_time, hnsw_serialization_time, reorganization_time);

    // expected = true;
    // rebuilding.compare_exchange_strong(expected, false); // atomic check and flip
    RebuildState exp = REBUILDING;
    bool swapped = rebuild_state.compare_exchange_strong(exp, DRAINING);
    assert(swapped); // should succeed: we must be REBUILDING here

    std::vector<std::pair<float*, int>> local_inserts;
    std::vector<int> local_deletes;
    {
        std::lock_guard<std::mutex> lg(insert_log_mutex_);
        local_inserts.swap(insert_log_);
    }
    {
        std::lock_guard<std::mutex> lg(delete_log_mutex_);
        local_deletes.swap(delete_log_);
    }

    std::vector<float> dummy_distances;
    std::atomic<size_t> dummy_completed{0};

    for (auto &entry : local_inserts) {
        // std::cout << "!!!DRAINING INSERTS!!!\n";
        float* insert_vector = entry.first;
        int label = entry.second;
        handle_insert(insert_vector, label, label, dummy_distances, dummy_completed);

        free(insert_vector); // allocated persistent copies when logging
    }

    for (int label : local_deletes) {
        std::cout << "!!!DRAINING DELETES!!!\n";
        handle_delete(label, world_size);
    }

    RebuildState exp2 = DRAINING;
    bool swapped2 = rebuild_state.compare_exchange_strong(exp2, IDLE);
    assert(swapped2);

    // {
    //     std::lock_guard<std::mutex> lock(insert_log_mutex);
    //     for (auto& entry : insert_log) {
    //         free(entry.first);
    //     }
    //     // std::vector<float> dummy_distances;
    //     // std::atomic<size_t> dummy_completed{0};
    //     // for (auto& entry : insert_log) {
    //     //     float* insert_vector = entry.first;
    //     //     int label = entry.second;
    //     //     handle_insert(
    //     //         insert_vector,
    //     //         label,
    //     //         label,
    //     //         dummy_distances, // no need to log distances during rebuild
    //     //         dummy_completed // no need to log completed during rebuild
    //     //     );
    //     //     free(entry.first);
    //     // }

    //     insert_log.clear();
    // }

    if (rebuild_pending_.exchange(false)) {
        return reBuild(world_size, full_threshold, partial_threshold);
    }

    if (rebuild_type == FULL_REBUILD_REQUEST)
        return 1;
    return 2;
}

// ── checkNeedRebuild ─────────────────────────────────────────────────────────
// Runs rePartition, caches the result, and returns the rebuild type without
// sending any MPI messages or modifying internal state.  Returns 0 if no
// rebuild is needed, 1 for a full rebuild, 2 for a partial rebuild.
// Must be followed by doRebuildSimple() when the return value is non-zero.
int Coordinator::checkNeedRebuild(int full_threshold, int partial_threshold,
                                   int ef_construction, int M_meta) {
    // Discard any stale cached rebuild from a previous call.
    if (cached_new_meta_HNSW_) {
        delete cached_new_meta_HNSW_;
        cached_new_meta_HNSW_ = nullptr;
    }
    cached_new_partitions_.clear();
    cached_hnsw_buffer_.clear();
    cached_rebuild_type_    = 0;
    cached_elements_moved_  = 0;
    cached_centers_moved_   = 0;
    cached_repart_hnsw_s_   = 0.0;
    cached_repart_bottom_s_ = 0.0;
    cached_repart_kaffpa_s_ = 0.0;
    cached_repart_relabel_s_= 0.0;

    if (full_threshold >= static_cast<int>(ncenters_) ||
        partial_threshold >= static_cast<int>(ncenters_)) {
        return 0;
    }
    hnswlib::HierarchicalNSW<float>* new_meta = nullptr;
    std::vector<int>                 new_parts;
    const int to_move = rePartition(new_parts, new_meta, ef_construction, M_meta,
                                    &cached_repart_hnsw_s_,
                                    &cached_repart_bottom_s_,
                                    &cached_repart_kaffpa_s_,
                                    &cached_repart_relabel_s_);

    // No rebuild needed: to_move is below both thresholds.
    // A threshold >= ncenters effectively disables that rebuild type.
    // A threshold of 0 triggers a rebuild on every check.
    if (to_move < full_threshold && to_move < partial_threshold) {
        delete new_meta;
        return 0;
    }

    // Count centers and elements that will physically move between shards.
    // new_parts is already relabeled by matchPartitions_ so it is directly
    // comparable with the current partitions[] array.
    {
        int centers = 0, elems = 0;
        for (size_t c = 0; c < new_parts.size(); c++) {
            if (new_parts[c] != partitions[c]) {
                centers++;
                elems += center_counts_[c];
            }
        }
        cached_centers_moved_  = centers;
        cached_elements_moved_ = elems;
    }

    // Cache materials needed for doRebuildSimple().
    cached_new_meta_HNSW_    = new_meta;
    cached_new_partitions_   = std::move(new_parts);

    cached_new_meta_HNSW_->saveIndex("tmp_hnsw.bin");
    {
        std::ifstream f("tmp_hnsw.bin", std::ios::binary);
        cached_hnsw_buffer_.assign(std::istreambuf_iterator<char>(f), {});
    }

    cached_rebuild_type_ = (to_move >= full_threshold) ? 1 : 2;
    return cached_rebuild_type_;
}

// ── doRebuildSimple ──────────────────────────────────────────────────────────
// Executes the rebuild prepared by checkNeedRebuild().  Sends the appropriate
// FULL/PARTIAL_REBUILD_REQUEST header + serialised meta-HNSW + new partitions
// to every executor, then blocks until each replies with REBUILD_SUCCESS.
// Finally swaps the new meta-HNSW and partitions into the Coordinator's
// internal state.  Safe only in serial (non-concurrent) contexts; it does not
// update rebuild_state or drain the insert/delete logs.
void Coordinator::doRebuildSimple(int world_size) {
    assert(cached_new_meta_HNSW_ != nullptr && "doRebuildSimple called without a cached rebuild");

    const MessageType rebuild_type = (cached_rebuild_type_ == 1)
                                     ? FULL_REBUILD_REQUEST
                                     : PARTIAL_REBUILD_REQUEST;
    const int meta_size = static_cast<int>(cached_hnsw_buffer_.size());

    // Send header + HNSW bytes + partitions to every executor.
    for (int i = 1; i < world_size; i++) {
        MessageHeader hdr;
        hdr.type = rebuild_type;
        hdr.size = static_cast<size_t>(meta_size);
        hdr.tag  = 0;
        MPI_Send(&hdr, sizeof(MessageHeader), MPI_BYTE, i, 0, MPI_COMM_WORLD);
        MPI_Send(cached_hnsw_buffer_.data(), meta_size, MPI_BYTE,
                 i, META_HNSW_SEND, MPI_COMM_WORLD);
        MPI_Send(cached_new_partitions_.data(), static_cast<int>(ncenters_), MPI_INT,
                 i, META_PARTITIONS_SEND, MPI_COMM_WORLD);
    }

    // Wait for REBUILD_SUCCESS from every executor.
    for (int i = 1; i < world_size; i++) {
        MessageHeader resp;
        MPI_Recv(&resp, sizeof(MessageHeader), MPI_BYTE,
                 i, REBUILD_SUCCESS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Swap in the new meta-HNSW and partitions.
    {
        std::unique_lock<std::shared_mutex> lk(graph_mutex_);
        if (meta_HNSW_) delete meta_HNSW_;
        meta_HNSW_ = cached_new_meta_HNSW_;
        partitions  = cached_new_partitions_;
    }

    // Clear the cache.
    cached_new_meta_HNSW_ = nullptr;
    cached_new_partitions_.clear();
    cached_hnsw_buffer_.clear();
    cached_rebuild_type_ = 0;
}

// ── doRebuildDelta ────────────────────────────────────────────────────────────
// Like doRebuildSimple() but sends INPLACE_REBUILD_REQUEST so each executor
// performs a mark-delete + insert pass instead of a full graph reconstruction.
// All other protocol details (meta-HNSW bytes + partition array) are identical.
void Coordinator::doRebuildDelta(int world_size) {
    assert(cached_new_meta_HNSW_ != nullptr && "doRebuildDelta called without a cached rebuild");

    const int meta_size = static_cast<int>(cached_hnsw_buffer_.size());

    for (int i = 1; i < world_size; i++) {
        MessageHeader hdr;
        hdr.type = INPLACE_REBUILD_REQUEST;
        hdr.size = static_cast<size_t>(meta_size);
        hdr.tag  = 0;
        MPI_Send(&hdr, sizeof(MessageHeader), MPI_BYTE, i, 0, MPI_COMM_WORLD);
        MPI_Send(cached_hnsw_buffer_.data(), meta_size, MPI_BYTE,
                 i, META_HNSW_SEND, MPI_COMM_WORLD);
        MPI_Send(cached_new_partitions_.data(), static_cast<int>(ncenters_), MPI_INT,
                 i, META_PARTITIONS_SEND, MPI_COMM_WORLD);
    }

    // Participate in the executor-to-executor AllToAll vector exchange.
    // Executors call MPI_Alltoall (counts) + 2x MPI_Alltoallv (vecs, labels)
    // over MPI_COMM_WORLD after their classify pass.  The coordinator has no
    // shard and sends/receives nothing, but must call the same three collectives
    // to keep all ranks in step — otherwise the collectives never complete.
    {
        // dummy_i / dummy_f: non-null stack pointers required by MPI even when
        // all counts are zero.  Their values are never read.
        int   dummy_i = 0;
        float dummy_f = 0.0f;
        std::vector<int> zero_counts(world_size, 0);
        std::vector<int> zero_displs(world_size, 0);

        // Round 1: element counts.
        std::vector<int> recv_counts(world_size, 0);
        MPI_Alltoall(zero_counts.data(), 1, MPI_INT,
                     recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        // recv_counts will all be 0: no executor sends vectors to rank 0.

        // Round 2: vectors.
        MPI_Alltoallv(&dummy_f, zero_counts.data(), zero_displs.data(), MPI_FLOAT,
                      &dummy_f, recv_counts.data(), zero_displs.data(), MPI_FLOAT,
                      MPI_COMM_WORLD);

        // Round 3: labels.
        MPI_Alltoallv(&dummy_i, zero_counts.data(), zero_displs.data(), MPI_INT,
                      &dummy_i, recv_counts.data(), zero_displs.data(), MPI_INT,
                      MPI_COMM_WORLD);
    }

    for (int i = 1; i < world_size; i++) {
        MessageHeader resp;
        MPI_Recv(&resp, sizeof(MessageHeader), MPI_BYTE,
                 i, REBUILD_SUCCESS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    {
        std::unique_lock<std::shared_mutex> lk(graph_mutex_);
        if (meta_HNSW_) delete meta_HNSW_;
        meta_HNSW_ = cached_new_meta_HNSW_;
        partitions  = cached_new_partitions_;
    }

    cached_new_meta_HNSW_ = nullptr;
    cached_new_partitions_.clear();
    cached_hnsw_buffer_.clear();
    cached_rebuild_type_ = 0;
}

void Coordinator::load_gp(const std::string& prefix, int ef_search) {
    std::string hnsw_path = prefix + ".pyramid_routing_index";
    std::string partitions_path = hnsw_path + ".routing_index_partition";

    std::cout << "[Coordinator] Loading HNSW from: " << hnsw_path << "\n";
    meta_HNSW_ = new hnswlib::HierarchicalNSW<float>(space_, hnsw_path);

    if (meta_HNSW_ == nullptr) {
        std::cerr << "[Coordinator] ERROR LOADING metaHNSW\n";
        return;
    }

    meta_HNSW_->setEf(ef_search);

    std::cout << "[Coordinator] Loading partitions from: " << partitions_path << "\n";
    std::ifstream in(partitions_path);
    std::vector<int> partition;
    int part;
    while (in >> part) partition.push_back(part);

    partitions = partition;
}

void Coordinator::load(const std::string& dir_path, int ef_search) {
    std::string hnsw_path = dir_path + "/metaHNSW.bin";
    std::string partitions_path = dir_path + "/partitions.bin";

    std::string centers_path = dir_path + "/centers_pos.bin";
    std::string centers_count_path = dir_path + "/centers_counts.bin";
    std::string centers_map_path = dir_path + "/labels_to_centers.bin";

    std::cout << "[Coordinator] Loading HNSW from: " << hnsw_path << "\n";
    meta_HNSW_ = new hnswlib::HierarchicalNSW<float>(space_, hnsw_path);

    if (meta_HNSW_ == nullptr) {
        std::cerr << "[Coordinator] ERROR LOADING metaHNSW\n";
        return;
    }

    meta_HNSW_->setEf(ef_search);

    std::cout << "[Coordinator] Loading partitions from: " << partitions_path << "\n";
    std::ifstream in(partitions_path, std::ios::binary);
    size_t size = 0;
    in.read(reinterpret_cast<char*>(&size), sizeof(size));
    partitions = std::vector<int>(size);
    in.read(reinterpret_cast<char*>(partitions.data()), size * sizeof(int));

    std::cout << "[Coordinator] size of partitions: " << size << "\n";

    this->ncenters_ = size;
    this->num_partitions_ = static_cast<size_t>(
        *std::max_element(partitions.begin(), partitions.end()) + 1
    );

    std::cout << "[Coordinator] Loading centers from: " << centers_path << "\n";
    std::ifstream in2(centers_path, std::ios::binary);
    centers_ = std::vector<float>(ncenters_ * dim_);
    in2.read(reinterpret_cast<char*>(centers_.data()), ncenters_ * dim_ * sizeof(float));

    std::cout << "[Coordinator] Loading center counts from: " << centers_count_path << "\n";
    std::ifstream in3(centers_count_path, std::ios::binary);
    center_counts_ = std::vector<int>(ncenters_);
    in3.read(reinterpret_cast<char*>(center_counts_.data()), ncenters_ * sizeof(int));

    std::cout << "[Coordinator] Loading label to center map from " << centers_map_path << "\n";
    std::ifstream in4(centers_map_path, std::ios::binary);
    std::size_t labels_size;
    in4.read(reinterpret_cast<char*>(&labels_size), sizeof(labels_size));

    label_to_center_.reserve(labels_size);
    for (std::size_t i = 0; i < labels_size; ++i) {
        int key, value;
        in4.read(reinterpret_cast<char*>(&key), sizeof(key));
        in4.read(reinterpret_cast<char*>(&value), sizeof(value));
        label_to_center_.emplace(key, value);
    }
}

void Coordinator::save(const std::string& output_dir) {
    std::shared_lock lock(graph_mutex_);
    std::cout << "[Coordinator] Saving to: " << output_dir << "\n";
    // make output dir if it doesn't exist
    std::filesystem::create_directories(output_dir);

    std::string hnsw_filename = output_dir + "/metaHNSW.bin";
    meta_HNSW_->saveIndex(hnsw_filename);

    std::string partition_filename = output_dir + "/partitions.bin";
    std::ofstream out(partition_filename, std::ios::binary);
    size_t size = partitions.size();
    out.write(reinterpret_cast<const char*>(&size), sizeof(size));
    out.write(reinterpret_cast<const char*>(partitions.data()), size * sizeof(int));

    std::string centers_filename = output_dir + "/centers_pos.bin";
    logFloatVec(centers_, centers_filename);

    std::string center_counts_filename = output_dir + "/centers_counts.bin";
    logIntVec(center_counts_, center_counts_filename);

    std::string labels_to_centers_filename = output_dir + "/labels_to_centers.bin";
    std::ofstream out_labels(labels_to_centers_filename, std::ios::binary);
    std::size_t size_labels = label_to_center_.size();
    out_labels.write(reinterpret_cast<const char*>(&size_labels), sizeof(size_labels));

    for (const auto& [key, value] : label_to_center_) {
        out_labels.write(reinterpret_cast<const char*>(&key), sizeof(key));
        out_labels.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }
}

std::vector<size_t> Coordinator::route_query(
    float* query_vector,
    size_t branching_factor,
    float* dist
) {
    return getPartitionsForSearch_Branching_(query_vector, static_cast<int>(branching_factor), dist);
}

std::vector<size_t> Coordinator::route_query_nprobe(
    float* query_vector,
    size_t nprobe,
    float* dist
) {
    return getPartitionsForSearch_Nprobe_(query_vector, static_cast<int>(nprobe), dist);
}

std::vector<size_t> Coordinator::route_query_recall_target(
    float* query_vector,
    float recall_target,
    float* dist
) {
    return getPartitionsForSearch_RecallTgt_(query_vector, recall_target, dist);
}

std::vector<std::vector<size_t>> Coordinator::route_queries(
    const std::vector<float>& query_vectors,
    RoutingMode mode,
    float param
) {
    const size_t num_queries = query_vectors.size() / dim_;
    std::vector<std::vector<size_t>> results(num_queries);

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_queries; i++) {
        float* vec = const_cast<float*>(query_vectors.data() + i * dim_);
        switch (mode) {
            case RoutingMode::BranchingFactor:
                results[i] = getPartitionsForSearch_Branching_(vec, static_cast<int>(param));
                break;
            case RoutingMode::NProbe:
                results[i] = getPartitionsForSearch_Nprobe_(vec, static_cast<int>(param));
                break;
            case RoutingMode::RecallTarget:
                results[i] = getPartitionsForSearch_RecallTgt_(vec, param);
                break;
        }
    }

    return results;
}

std::vector<int> Coordinator::handle_query(
    float* query_vector,
    int query_idx,
    size_t num_neighbors,
    RoutingMode mode,
    float param,
    std::vector<std::atomic<int>>* executor_hits
) {
    if (!comm_) {
        throw std::runtime_error("[Coordinator] communicator not set");
    }

    std::vector<size_t> executors;

    switch (mode) {
        case RoutingMode::BranchingFactor: {
            executors = route_query(query_vector, param);
            break;
        }
        case RoutingMode::NProbe: {
            executors = route_query_nprobe(query_vector, param);
            break;
        }
        case RoutingMode::RecallTarget: {
            executors = route_query_recall_target(query_vector, param);
            break;
        }
    }
     
    if (executor_hits) {
        for (size_t idx : executors) (*executor_hits)[idx]++;
    }

    int tag = make_tag(QUERY_SEND, query_idx);
    for (size_t idx : executors) {
        MessageHeader header;
        header.type = QUERY_SEND;
        header.size = num_neighbors;
        header.tag = tag;

        comm_->send_header(header, idx + 1);
        comm_->send_vector(query_vector, dim_, idx + 1, tag);
    }

    std::vector<int> results(executors.size() * num_neighbors);
    for (size_t i = 0; i < executors.size(); ++i) {
        size_t idx = executors[i];
        int* dest = results.data() + i * num_neighbors;
        comm_->recv_result(dest, num_neighbors, idx + 1, tag);
    }

    return results;
}

// void Coordinator::handle_query(
//     float* query_vector,
//     int query_idx,
//     const std::vector<int>& ground_truth_idx,
//     size_t num_neighbors,
//     size_t branching_factor,
//     size_t num_closest_from_ground,
//     std::vector<double>& latencies,
//     std::atomic<size_t>& correct,
//     std::atomic<size_t>& completed,
//     std::vector<int>* access_rate,
//     std::vector<std::atomic<int>>* executor_hits,
//     float* recall
// ) {
//     std::vector<size_t> executors = getPartitionsFor(query_vector, branching_factor);
//     if (access_rate != nullptr && executor_hits != nullptr) {
//         (*access_rate)[query_idx] = executors.size();
//         for (size_t idx : executors) (*executor_hits)[idx]++;
//     }

//     int results[executors.size()][num_neighbors];
//     int tag = make_tag(QUERY_SEND, query_idx);
//     // int tag = (rand() % 10000) + query_idx + 1;
//     double start = MPI_Wtime();
//     for (size_t idx : executors) {
//         MessageHeader header; header.type = QUERY_SEND;
//         header.size = num_neighbors;
//         header.tag = tag;
//         comm_.send_header(header, idx + 1);
//         comm_.send_vector(query_vector, dim_, idx + 1, tag);
//     }

//     for (size_t i = 0; i < executors.size(); ++i) {
//         size_t idx = executors[i];
//         comm_.recv_result(results[i], num_neighbors, idx + 1, tag);
//     }
    
//     double latency = MPI_Wtime() - start;
//     latencies[query_idx] = latency;

//     int relevant = 0;
//     for (size_t i = 0; i < executors.size(); ++i) {
//         for (int j = 0; j < num_neighbors; j++) {
//             for (int a = 0; a < num_closest_from_ground; a++) {
//                 if (results[i][j] == ground_truth_idx[a]) {
//                     relevant++;
//                 }
//             }
//         }
//     }

//     if (recall != nullptr)
//         *(recall) = relevant / float(num_closest_from_ground);

//     completed++;
// }

std::vector<std::vector<int>> Coordinator::handle_queries(
    const std::vector<float>& query_vectors,
    size_t num_neighbors,
    size_t branching_factor
) {
    if (!comm_) {
        throw std::runtime_error("[Coordinator] communicator not set");
    }

    std::vector<std::vector<float>> per_executor_batches(num_partitions_ + 1);
    std::vector<std::vector<int>> per_executor_queryid(num_partitions_ + 1);

    // make per-executor query batches
    # pragma omp parallel for
    for (size_t i = 0; i < query_vectors.size() / dim_; i++) {
        float* query_vector = const_cast<float*>(query_vectors.data() + i * dim_);
        std::vector<size_t> partitions = route_query(query_vector, branching_factor);
        for (size_t idx : partitions) {
            #pragma omp critical
            {
                per_executor_batches[idx + 1].insert(per_executor_batches[idx + 1].end(), query_vector, query_vector + dim_);
                per_executor_queryid[idx + 1].push_back(i);
            }
        }
    }

    int num_queries = query_vectors.size() / dim_;
    std::vector<std::vector<int>> per_query_results(num_queries);
    // send & aggregate results
    # pragma omp parallel for
    for (size_t idx = 1; idx <= num_partitions_; idx++) {
        if (per_executor_batches[idx].size() == 0) continue;

        MessageHeader header;
        header.type = QUERY_BATCH_SEND;
        header.size = per_executor_batches[idx].size() / dim_;
        header.tag = make_tag(QUERY_BATCH_SEND, idx);

        comm_->send_header(header, idx);
        comm_->send_vector_batch(per_executor_batches[idx].data(), per_executor_batches[idx].size() / dim_, dim_, idx, header.tag);

        std::vector<int> results(per_executor_queryid[idx].size() * num_neighbors);
        comm_->recv_result_batch(results.data(), per_executor_queryid[idx].size() * num_neighbors, idx, header.tag);

        for (size_t i = 0; i < per_executor_queryid[idx].size(); i++) {
            int query_id = per_executor_queryid[idx][i];
            per_query_results[query_id].insert(per_query_results[query_id].end(), results.data() + i * num_neighbors, results.data() + (i + 1) * num_neighbors);
        }
    }

    return per_query_results;
}

void Coordinator::handle_insert(
    float* insert_vector,
    int label,
    int insert_idx,
    std::vector<float>& insert_distances,
    std::atomic<size_t>& completed
) {
    if (!comm_) {
        throw std::runtime_error("[Coordinator] communicator not set");
    }

    RebuildState cur = rebuild_state.load(std::memory_order_acquire);
    if (cur == REBUILDING) {
        std::lock_guard<std::mutex> lock(insert_log_mutex_);
        float* persistent_vec = new float[dim_];
        memcpy(persistent_vec, insert_vector, sizeof(float) * dim_);
        insert_log_.push_back({persistent_vec, label});
        return;
    }

    // if (rebuilding) { // atomic load and check?
    //     {
    //         std::lock_guard<std::mutex> lock(insert_log_mutex);
    //         // std::cout << "[Coordinator] REBUILDING, logging insert " << insert_idx << "\n";
    //         float* persistent_vec = new float[dim_];
    //         memcpy(persistent_vec, insert_vector, sizeof(float) * dim_);
    //         insert_log.push_back({persistent_vec, label});
    //         return;
    //     }
    // }
    size_t executor;
    if (insert_distances.size() == 0 )
        executor = getPartitionsForInsert_(insert_vector, label, nullptr)[0];
    else 
        executor = getPartitionsForInsert_(insert_vector, label, &(insert_distances[insert_idx]))[0];
    
    // int tag = insert_idx + 1;
    int tag = make_tag(INSERT_SEND, insert_idx);
    MessageHeader header;
    header.type = INSERT_SEND;
    header.tag = tag;
    
    comm_->send_header(header, executor + 1);
    comm_->send_insert(insert_vector, dim_, label, executor + 1, tag);

    bool success = comm_->recv_ack(INSERT_SUCCESS, executor + 1, tag);
    if (!success) {
        std::cout << "FAILED TO INSERT " << insert_idx << "\n";
    }

    completed++;
}

void Coordinator::handle_inserts(std::vector<float>& insert_vectors, std::vector<int>& labels) {
    if (!comm_) {
        throw std::runtime_error("[Coordinator] communicator not set");
    }

    RebuildState cur = rebuild_state.load(std::memory_order_acquire);

    if (cur == REBUILDING) {
        std::lock_guard<std::mutex> lock(insert_log_mutex_);
        float* persistent_vecs = new float[insert_vectors.size()];
        memcpy(persistent_vecs, insert_vectors.data(), sizeof(float) * insert_vectors.size());
        for (size_t i = 0; i < labels.size(); i++) {
            insert_log_.push_back({persistent_vecs + i * dim_, labels[i]});
        }
        return;
    }

    std::vector<std::vector<float>> per_executor_vecs(num_partitions_ + 1);
    std::vector<std::vector<int>> per_executor_labels(num_partitions_ + 1);

    // make per-executor insert batches
    #pragma omp parallel for
    for (size_t i = 0; i < labels.size(); i++) {
        float* insert_vector = insert_vectors.data() + i * dim_;
        int label = labels[i];
        size_t partition = getPartitionsForInsert_(insert_vector, label, nullptr)[0];

        #pragma omp critical
        {
            per_executor_vecs[partition + 1].insert(per_executor_vecs[partition + 1].end(), insert_vector, insert_vector + dim_);
            per_executor_labels[partition + 1].push_back(label);
        }
    }

    // send batches
    #pragma omp parallel for
    for (size_t idx = 1; idx <= num_partitions_; idx++) {
        if (per_executor_labels[idx].size() == 0) continue;

        int tag = make_tag(INSERT_BATCH_SEND, idx);
        MessageHeader header;
        header.type = INSERT_BATCH_SEND;
        header.size = per_executor_labels[idx].size();
        header.tag = tag;

        comm_->send_header(header, idx);
        comm_->send_insert_batch(
            per_executor_vecs[idx].data(), 
            per_executor_labels[idx].data(),
            per_executor_labels[idx].size(), 
            dim_, 
            idx, 
            tag
        );

        bool success = comm_->recv_ack(INSERT_BATCH_SUCCESS, idx, tag);
        if (!success) {
            std::cout << "FAILED TO INSERT BATCH FOR Executor " << idx << "\n";
        }
    }
}

bool Coordinator::handle_delete(int label, int world_size) {
    if (!comm_) {
        throw std::runtime_error("[Coordinator] communicator not set");
    }

    // if (label % 10000 == 0)
    //     std::cout << "[Coordinator] Handling delete for label " << label << "\n";
    RebuildState cur = rebuild_state.load(std::memory_order_acquire);
    if (cur == REBUILDING) {
        std::lock_guard<std::mutex> lg(delete_log_mutex_);
        delete_log_.push_back(label);
        return true;
    }

    unsigned int tag = make_tag(DELETE_SEND, label);

    bool success = false;
    std::vector<float> vec(dim_);
    int center_idx = label_to_center_[label];
    int partition = partitions[center_idx];
    comm_->send_delete(label, partition + 1, tag);
    if (comm_->recv_ack(DELETE_SUCCESS, partition + 1, tag)) {
        success = true;
        comm_->recv_delete(vec.data(), dim_, partition + 1, tag);
    }

    if (success)
        updateCentersForDelete_(vec, center_idx);

    return success;
}

void Coordinator::handle_deletes(const std::vector<int>& labels, int world_size) {
    if (!comm_) {
        throw std::runtime_error("[Coordinator] communicator not set");
    }

    RebuildState cur = rebuild_state.load(std::memory_order_acquire);
    if (cur == REBUILDING) {
        std::lock_guard<std::mutex> lg(delete_log_mutex_);
        delete_log_.insert(delete_log_.end(), labels.begin(), labels.end());
        return;
    }

    std::vector<std::vector<int>> per_executor_labels(num_partitions_ + 1);
    std::vector<std::vector<int>> per_executor_centerids(num_partitions_ + 1);
    #pragma omp parallel for
    for (size_t i = 0; i < labels.size(); i++) {
        int label = labels[i];
        int center = label_to_center_[label];
        size_t partition = partitions[center];
        #pragma omp critical
        {
            per_executor_labels[partition + 1].push_back(label);
            per_executor_centerids[partition + 1].push_back(center);
        }
    }

    // send batches
    std::vector<std::vector<float>> per_executor_vecs(num_partitions_ + 1);
    #pragma omp parallel for
    for (size_t idx = 1; idx <= num_partitions_; idx++) {
        if (per_executor_labels[idx].size() == 0) continue;
        unsigned int tag = make_tag(DELETE_BATCH_SEND, idx);
        MessageHeader header;
        header.type = DELETE_BATCH_SEND;
        header.size = per_executor_labels[idx].size();
        header.tag = tag;
        
        comm_->send_header(header, idx);
        comm_->send_delete_batch(per_executor_labels[idx].data(), per_executor_labels[idx].size(), idx, tag);
        bool success = comm_->recv_ack(DELETE_BATCH_SUCCESS, idx, tag);
        if (!success) {
            std::cout << "FAILED TO DELETE BATCH FOR Executor " << idx << "\n";
        }

        comm_->recv_delete_batch_results(per_executor_vecs[idx].data(), dim_, idx, tag);

        updateCentersForDeleteBatch(per_executor_vecs[idx], per_executor_centerids[idx]);
        // two options: run in the parallel loop per executor, 
        // or aggregate all vectors and run a single updateCentersForDeleteBatch after the loop.
        // - requires sorting received vectors by label
    }
    
}

    // // Random initialization of centers
    // std::vector<size_t> indices(nPrime);
    // std::iota(indices.begin(), indices.end(), 0);
    // std::shuffle(indices.begin(), indices.end(), gen);

    // for (size_t c = 0; c < m_centers; ++c) {
    //     std::copy(sample + indices[c] * dim_, sample + (indices[c] + 1) * dim_, centers + c * dim_);
    // }


int Coordinator::kmeans_(float* sample, size_t nPrime, size_t m_centers, float* centers_, float EPSILON) {
    center_counts_ = std::vector<int>(m_centers, 0);
    int iters = kmeans(
        sample,
        nPrime,
        dim_,
        m_centers,
        centers_,
        center_counts_.data(),
        EPSILON,
        KMEANS_EPOCHS
    );
    // Reset counts to zero so subsequent inserts accumulate from a clean
    // baseline, matching the reference pipeline (msturing-cluster-analysis.cpp).
    std::fill(center_counts_.begin(), center_counts_.end(), 0);
    return iters;
}

// ========== Executor Implementation ==========
Executor::Executor(int node_id, int dim, Communicator& comm, Log* logger)
    : node_id_(node_id),
      dim_(dim),
      comm_(comm),
      data_(nullptr),
      data_count_(0)
{
    if (logger) logger_ = logger;
    space_ = new hnswlib::L2Space(dim_);
    computeDistance_ = [this](float* a, float* b) { return computeEuclideanDistance(a, b, dim_); };
}

void Executor::setData(float* data, int* indices, size_t count) {
    data_ = data;
    indices_ = indices;
    data_count_ = count;
    std::cout << "[Executor " << node_id_ << " ] Data set with " << count << " elements\n";
}

void Executor::receiveData(size_t nrecv_vecs) {
    size_t required_nvecs = data_count_ + nrecv_vecs;
    if (local_vectors_.capacity() < required_nvecs * dim_)
        local_vectors_.reserve(required_nvecs * dim_ * 2);
    if (local_indices_.capacity() < required_nvecs)
        local_indices_.reserve(required_nvecs * 2);

    // std::cout << "[Executor " << node_id_ << "] receiveData() reserved\n ";

    local_vectors_.resize(required_nvecs * dim_);
    local_indices_.resize(required_nvecs);

    // std::cout << "[Executor " << node_id_ << "]--receiveData()-- Receiving " << nrecv_vecs << " vectors from coordinator\n";

    float* vec_recv_ptr = local_vectors_.data() + (data_count_ * dim_);
    int* idx_recv_ptr = local_indices_.data() + data_count_;

    comm_.receive_vector_data(vec_recv_ptr, 
                              idx_recv_ptr,
                              nrecv_vecs, dim_);

    data_count_ += nrecv_vecs;

    data_ = local_vectors_.data();
    indices_ = local_indices_.data();
}

void Executor::setEfSearch(int ef_search) {
    std::unique_lock lock(graph_mutex_);
    sub_HNSW_->setEf(ef_search);
}

void Executor::build(
    int ef_construction,
    int M_sub,
    int num_building_threads
) {
    if (data_ == nullptr) {
        std::cerr << "[Executor " <<  node_id_ << "]: set data before building\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (num_building_threads == -1)
        num_building_threads = omp_get_max_threads();

    std::cout << "[Executor " << node_id_ << " ] Building local sub-index\n";
    double start = MPI_Wtime();
    float* norm_element = new float[dim_];
    sub_HNSW_ = new hnswlib::HierarchicalNSW<float>(space_, data_count_, M_sub, ef_construction, 100, true);
    sub_HNSW_->addPoint(data_, indices_[0], /*replace_deleted=*/true); // first element not thread safe
    omp_set_num_threads(num_building_threads);
    std::cout << "[Executor " << node_id_ << "] - num threads building index: " << num_building_threads << "/"  << omp_get_max_threads() << "\n";
    #pragma omp parallel for num_threads(num_building_threads)
    for (int j = 1; j < data_count_; j++) {
        sub_HNSW_->addPoint(data_ + j*dim_, indices_[j], /*replace_deleted=*/true);
    }

    delete[] norm_element;
    double end = MPI_Wtime();
    logger_->index_build_time = end - start;

    sub_HNSW_->setEf(ef_construction);
}

void Executor::partialReBuild(
    int meta_size, 
    int ncenters, 
    int world_size,
    int ef_construction,
    int num_building_threads
) {
    std::cout << "[Executor " << node_id_ << "] partial rebuild\n";
    if (num_building_threads == -1)
        num_building_threads = omp_get_max_threads();

    // Step 1: Gather active local vectors
    std::vector<float*> active_vectors;
    std::vector<int> active_indices;

    size_t nelts = sub_HNSW_->getCurrentElementCount();
    for (hnswlib::tableint internal_id = 0; internal_id < nelts; internal_id++) {
        if (!sub_HNSW_->isMarkedDeleted(internal_id)) {
            float* vec_ptr = reinterpret_cast<float*>(sub_HNSW_->getDataByInternalId(internal_id));
            size_t external_label = sub_HNSW_->getExternalLabel(internal_id);
            active_vectors.push_back(vec_ptr);
            active_indices.push_back(external_label);
        }
    }

    // Step 2: Receive meta graph
    // std::cout << "[Executor " << node_id_ << "] receiving meta HNSW of size " << meta_size << "\n";
    std::vector<char> buffer(meta_size);
    MPI_Recv(buffer.data(), meta_size, MPI_BYTE, 0, META_HNSW_SEND, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // std::cout << "[Executor " << node_id_ << "] RECEIVED meta HNSW\n";

    std::ofstream outfile("tmp_hnsw_received.bin", std::ios::binary);
    outfile.write(buffer.data(), meta_size);
    outfile.close();

    hnswlib::HierarchicalNSW<float>* metaHNSW =
        new hnswlib::HierarchicalNSW<float>(space_, "tmp_hnsw_received.bin");

    // Step 3: Receive partition mapping
    std::vector<int> partitions(ncenters);
    MPI_Recv(partitions.data(), ncenters, MPI_INT, 0, META_PARTITIONS_SEND, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // std::cout << "[Executor " << node_id_ << "] RECEIVED partitions\n";

    // Step 4: Prepare outbound buffers
    std::vector<std::vector<float>> partition_vectors(world_size);
    std::vector<std::vector<int>>   partition_indices(world_size);

    // Step 5: Iterate local elements — decide where they go
    // std::cout << "[Executor " << node_id_ << "] partitioning local data\n";
    #pragma omp parallel for num_threads(num_building_threads)
    for (int i = 0; i < (int)active_vectors.size(); i++) {
        float* vec = active_vectors[i];
        int index = active_indices[i];
        hnswlib::labeltype label = metaHNSW->searchKnn(vec, 1).top().second;
        int p = partitions[label] + 1;  // executor ID for this point
        // std::cout << "[Executor " << node_id_ << "] mid partitioning - " << i << "\n";
        if (p != node_id_) {
            #pragma omp critical(rebuild_partitions)
            {
                partition_vectors[p].insert(partition_vectors[p].end(), vec, vec + dim_);
                partition_indices[p].push_back(index);
            }
        }
    }
    // std::cout << "[Executor " << node_id_ << "] DONE partitioning local data\n";

    // Step 6: Exchange counts with peers
    std::vector<int> num_to_recv(world_size, 0);
    std::vector<int> num_to_send(world_size);
    for (int i = 1; i < world_size; i++) {
        num_to_send[i] = partition_vectors[i].size() / dim_;
    }

    // std::cout << "[Executor " << node_id_ << "] exchanging N\n";
    std::vector<MPI_Request> requests;
    for (int peer = 1; peer < world_size; ++peer) {
        if (peer == node_id_) continue;
        requests.emplace_back();
        MPI_Irecv(&num_to_recv[peer], 1, MPI_INT, peer, TAG_SEND_NUM, MPI_COMM_WORLD, &requests.back());
        requests.emplace_back();
        MPI_Isend(&num_to_send[peer], 1, MPI_INT, peer, TAG_SEND_NUM, MPI_COMM_WORLD, &requests.back());
    }
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    requests.clear();
    // std::cout << "[Executor " << node_id_ << "] DONE exchanging N\n";

    // Step 7: Prepare receive buffers
    int total_recv = std::accumulate(num_to_recv.begin(), num_to_recv.end(), 0);
    std::vector<float> all_recv_vectors(total_recv * dim_);
    std::vector<int>   all_recv_labels(total_recv);

    std::vector<int> offset(world_size, 0);
    for (int i = 1; i < world_size; ++i)
        offset[i] = offset[i - 1] + num_to_recv[i - 1];

    // Step 8: Exchange actual vector data
    // std::cout << "[Executor " << node_id_ << "] exchanging vecs\n";
    for (int peer = 1; peer < world_size; ++peer) {
        if (peer == node_id_) continue;

        int n_recv = num_to_recv[peer];
        int peer_offset = offset[peer];

        if (n_recv > 0) {
            requests.emplace_back();
            MPI_Irecv(all_recv_vectors.data() + peer_offset * dim_, n_recv * dim_, MPI_FLOAT, peer, TAG_SEND_VECS, MPI_COMM_WORLD, &requests.back());
            requests.emplace_back();
            MPI_Irecv(all_recv_labels.data() + peer_offset, n_recv, MPI_INT, peer, TAG_SEND_LABELS, MPI_COMM_WORLD, &requests.back());
        }

        if (num_to_send[peer] > 0) {
            requests.emplace_back();
            MPI_Isend(partition_vectors[peer].data(), num_to_send[peer] * dim_, MPI_FLOAT, peer, TAG_SEND_VECS, MPI_COMM_WORLD, &requests.back());
            requests.emplace_back();
            MPI_Isend(partition_indices[peer].data(), num_to_send[peer], MPI_INT, peer, TAG_SEND_LABELS, MPI_COMM_WORLD, &requests.back());
        }
    }
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    // std::cout << "[Executor " << node_id_ << "] DONE exchanging vecs\n";

    // Step 9: Add received points into existing HNSW
    size_t next_data_count = sub_HNSW_->getCurrentElementCount() + total_recv;
    if (next_data_count > sub_HNSW_->getMaxElements())
        sub_HNSW_->resizeIndex(next_data_count);

    #pragma omp parallel for num_threads(num_building_threads)
    for (int i = 0; i < total_recv; i++) {
        float* vec = all_recv_vectors.data() + (i * dim_);
        int index = all_recv_labels[i];
        sub_HNSW_->addPoint(vec, index, /*replace_deleted=*/true);
    }

    sub_HNSW_->setEf(ef_construction);

    // Step 10: Notify completion
    MessageHeader header; 
    header.type = REBUILD_SUCCESS;
    // // additionally send back num levels in the graph
    header.size = sub_HNSW_->maxlevel_;
    MPI_Send(&header, sizeof(MessageHeader), MPI_BYTE, 0, REBUILD_SUCCESS, MPI_COMM_WORLD);
}

void Executor::reBuild(
    int meta_size, 
    int ncenters, 
    int world_size, 
    int ef_construction, 
    int M_sub,
    int num_building_threads
) {
    std::cout << "[Executor " << node_id_ << "] rebuilding index \n";
    if (num_building_threads == -1)
        num_building_threads = omp_get_max_threads();

    std::vector<float*> active_vectors;
    std::vector<int> active_indices;

    // TODO: multithreading possible
    // TODO: ensure all pending inserts complete
    size_t active_count = 0;
    {
        size_t nelts = sub_HNSW_->getCurrentElementCount();
        for (hnswlib::tableint internal_id = 0; internal_id < nelts; internal_id++) {
            if (!sub_HNSW_->isMarkedDeleted(internal_id)) {
                float* vec_ptr = reinterpret_cast<float*>(sub_HNSW_->getDataByInternalId(internal_id));
                size_t external_label = sub_HNSW_->getExternalLabel(internal_id);
                active_vectors.push_back(vec_ptr);
                active_indices.push_back(external_label);
                active_count++;
            }
        }
    }

    std::vector<char> buffer(meta_size);
    MPI_Recv(buffer.data(), meta_size, MPI_BYTE, 0, META_HNSW_SEND, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Wrap buffer in stream and load
    std::ofstream outfile("tmp_hnsw_received.bin", std::ios::binary);
    outfile.write(buffer.data(), meta_size);
    outfile.close();

    hnswlib::HierarchicalNSW<float>* metaHNSW = new hnswlib::HierarchicalNSW<float>(space_, "tmp_hnsw_received.bin");

    std::vector<int> partitions(ncenters);
    MPI_Recv(partitions.data(), ncenters, MPI_INT, 0, META_PARTITIONS_SEND, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    hnswlib::HierarchicalNSW<float>* next_sub_HNSW = new hnswlib::HierarchicalNSW<float>(space_, active_count, M_sub, ef_construction, 100, true);

    std::vector<std::vector<float>> partition_vectors(world_size);
    std::vector<std::vector<int>> partition_indices(world_size);

    double start_iter = MPI_Wtime();
    // check which elements stay and move
    bool first_inserted = false;
    int idx = 0;
    int num_kept;
    while (!first_inserted && idx < active_count) { // first insert into HNSW must be sequential
        float* vec = active_vectors[idx];
        //  data_+ (idx * dim_);
        int index = active_indices[idx];
        hnswlib::labeltype label = metaHNSW->searchKnn(vec, 1).top().second;
        int p = partitions[label] + 1;
        if (p == node_id_) {
            first_inserted = true;
            next_sub_HNSW->addPoint(vec, index, /*replace_deleted=*/true);
            num_kept++;
        } else {
            partition_vectors[p].insert(partition_vectors[p].end(), vec, vec + dim_);
            partition_indices[p].push_back(index);
        }
        idx++;
    }

    // std::cout << "[Executor " << node_id_ << "] rebuild -- starting parallel iteration\n";
    #pragma omp parallel for num_threads(num_building_threads)
    for (int i = idx; i < active_count; i++) {
        float* vec = active_vectors[i];
        // data_ + (i * dim_);
        int index = active_indices[i];
        hnswlib::labeltype label = metaHNSW->searchKnn(vec, 1).top().second;
        int p = partitions[label] + 1;
        if (p == node_id_) {
            next_sub_HNSW->addPoint(vec, index, /*replace_deleted=*/true);
            num_kept++;
        }
        else {
            // std::cout << "[Executor " << node_id_ << "] rebuild -- entering critical region\n";
            #pragma omp critical(rebuild_partitions)
            {
                partition_vectors[p].insert(partition_vectors[p].end(), vec, vec + dim_);
                partition_indices[p].push_back(index);
            }
            // std::cout << "[Executor " << node_id_ << "] rebuild -- out of critical region\n";
        }
    }
    double end_iter = MPI_Wtime();

    std::cout << "[Executor " << node_id_ << "] iterate time : " <<  end_iter - start_iter << "\n";
    // std::cout << "[Executor " << node_id_ << "] num_kept : " <<  num_kept << "/" << active_count << "\n";

    std::vector<int> num_to_recv(world_size, 0);
    std::vector<int> num_to_send(world_size);
    for (int i = 1; i < world_size; i++) {
        num_to_send[i] = partition_vectors[i].size() / dim_;
    }

    // std::cout << "[Executor " << node_id_ << " ] exchanging nums \n";
    double start_exchange = MPI_Wtime();

    std::vector<MPI_Request> requests;
    for (int peer = 1; peer < world_size; ++peer) {
        if (peer == node_id_) continue;
        // --- RECV: n ---
        requests.emplace_back();
        MPI_Irecv(&num_to_recv[peer], 1, MPI_INT, peer, TAG_SEND_NUM, MPI_COMM_WORLD, &requests.back());

        // --- SEND: n ---
        requests.emplace_back();
        MPI_Isend(&num_to_send[peer], 1, MPI_INT, peer, TAG_SEND_NUM, MPI_COMM_WORLD, &requests.back());
    }

    // Wait for all n values
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    requests.clear();


    int total_recv = std::accumulate(num_to_recv.begin(), num_to_recv.end(), 0);
    std::vector<float> all_recv_vectors(total_recv * dim_);
    std::vector<int> all_recv_labels(total_recv);

    std::vector<int> offset(world_size, 0);
    for (int i = 1; i < world_size; ++i)
        offset[i] = offset[i - 1] + num_to_recv[i - 1];

    for (int peer = 1; peer < world_size; ++peer) {
        if (peer == node_id_) continue;

        if (num_to_recv[peer] > 0) {
            int n_recv = num_to_recv[peer];
            int peer_offset = offset[peer];

            // --- RECV: vectors and labels ---
            requests.emplace_back();
            MPI_Irecv(all_recv_vectors.data() + peer_offset * dim_, n_recv * dim_, MPI_FLOAT, peer, TAG_SEND_VECS, MPI_COMM_WORLD, &requests.back());
            requests.emplace_back();
            MPI_Irecv(all_recv_labels.data() + peer_offset, n_recv, MPI_INT, peer, TAG_SEND_LABELS, MPI_COMM_WORLD, &requests.back());
        }

        if (num_to_send[peer] > 0) {
            // --- SEND: vectors and labels ---
            requests.emplace_back();
            MPI_Isend(partition_vectors[peer].data(), num_to_send[peer] * dim_, MPI_FLOAT, peer, TAG_SEND_VECS, MPI_COMM_WORLD, &requests.back());
            requests.emplace_back();
            MPI_Isend(partition_indices[peer].data(), num_to_send[peer], MPI_INT, peer, TAG_SEND_LABELS, MPI_COMM_WORLD, &requests.back());
        }
    }

    // wait for all exchanges to happen
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    double end_exchange = MPI_Wtime();
    // std::cout << "[Executor " << node_id_ << " ] DONE EXCHANGING*******\n";
    std::cout << "[Executor " << node_id_ << "] rebuild exchange time : " <<  end_exchange - start_exchange << "\n";

    size_t next_data_count = next_sub_HNSW->getCurrentElementCount() + total_recv;
    if (next_data_count > next_sub_HNSW->getMaxElements()) 
        next_sub_HNSW->resizeIndex(next_data_count);


    double start_hnsw = MPI_Wtime();
    #pragma omp parallel for num_threads(num_building_threads)
    for (int i = 0; i < total_recv; i++) {
        float* vec = all_recv_vectors.data() + (i * dim_);
        int index = all_recv_labels[i];
        next_sub_HNSW->addPoint(vec, index, /*replace_deleted=*/true);
    }
    double end_hnsw = MPI_Wtime();

    std::cout << "[Executor " << node_id_ << "] rebuild hnsw time : " <<  end_hnsw - start_hnsw << "\n";
    // std::cout << "[Executor " << node_id_ << "] total num elements: " <<  next_data_count << "\n";

    next_sub_HNSW->setEf(ef_construction);

    // lock as you make the switch
    {   std::unique_lock lock(graph_mutex_);

        free(sub_HNSW_);
        sub_HNSW_ = next_sub_HNSW;
    }

    // Cache per-phase timings before notifying coordinator (so getters are
    // valid as soon as reBuild() returns on the experiment side).
    last_rebuild_iterate_s_  = end_iter     - start_iter;
    last_rebuild_exchange_s_ = end_exchange - start_exchange;
    last_rebuild_graph_s_    = end_hnsw     - start_hnsw;

    MessageHeader header; header.type = REBUILD_SUCCESS;
    MPI_Send(&header, sizeof(MessageHeader), MPI_BYTE, 0, REBUILD_SUCCESS, MPI_COMM_WORLD);
}

// ── reBuildDelta ─────────────────────────────────────────────────────────────
// In-place rebuild: instead of constructing a fresh sub-HNSW, this method
// (1) classifies every live element under the new partitions,
// (2) mark-deletes elements that migrated away and sends them to their new
//     owners via the same peer-exchange protocol as reBuild(),
// (3) inserts the elements received from other workers using replace_deleted=true
//     so that freed slots are preferentially reused, and
// (4) logs / stores the count of deleted slots that were not reused (i.e. the
//     number of departing elements whose graph positions remain as tombstones).
//
// Requires sub_HNSW_ to have been built with allow_replace_deleted=true,
// which is always the case for indexes created by Executor::build().
void Executor::reBuildDelta(
    int meta_size,
    int ncenters,
    int world_size,
    int num_building_threads
) {
    std::cout << "[Executor " << node_id_ << "] delta rebuild (in-place)\n";
    if (num_building_threads == -1)
        num_building_threads = omp_get_max_threads();

    // ── Step 1: receive new meta-HNSW and partition assignments ──────────────
    std::vector<char> buf(meta_size);
    MPI_Recv(buf.data(), meta_size, MPI_BYTE, 0, META_HNSW_SEND,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    const std::string tmp_path =
        "tmp_hnsw_delta_r" + std::to_string(node_id_) + ".bin";
    {
        std::ofstream f(tmp_path, std::ios::binary);
        f.write(buf.data(), meta_size);
    }
    hnswlib::HierarchicalNSW<float>* metaHNSW =
        new hnswlib::HierarchicalNSW<float>(space_, tmp_path);

    std::vector<int> partitions(ncenters);
    MPI_Recv(partitions.data(), ncenters, MPI_INT, 0, META_PARTITIONS_SEND,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // ── Step 2: classify live elements; collect those migrating away ──────────
    // Read pass: shared lock so no concurrent resize can occur.
    std::vector<std::vector<float>> send_vecs(world_size);
    std::vector<std::vector<int>>   send_labels(world_size);

    double start_iter = MPI_Wtime();
    {
        std::shared_lock<std::shared_mutex> lk(graph_mutex_);
        const size_t nelts = sub_HNSW_->getCurrentElementCount();

        #pragma omp parallel for schedule(dynamic) num_threads(num_building_threads)
        for (hnswlib::tableint iid = 0; iid < static_cast<hnswlib::tableint>(nelts); iid++) {
            if (sub_HNSW_->isMarkedDeleted(iid)) continue;

            int ext_lbl = static_cast<int>(sub_HNSW_->getExternalLabel(iid));

            // Canonical-ownership check: verify that label_lookup_ maps this
            // label back to this exact internal ID.  A mismatch means the slot
            // holds a "ghost" label — its label_lookup_ entry was already
            // updated to point to a different (live) slot by a replace_deleted
            // addPoint, but the stale label value was never cleared from the
            // data buffer.  Visiting ghost slots would feed the wrong label into
            // the mark-delete pass, potentially causing "markDelete: Label not
            // found" if the live entry was subsequently erased.
            //
            // No lock is needed here: the rebuild protocol guarantees no
            // concurrent writes to label_lookup_ during this iterate pass
            // (insertLocalBatch requires exclusive graph_mutex_, which is
            // blocked by our shared_lock above; no other shard touches this
            // local map).  All OMP threads are read-only, so concurrent reads
            // of the unordered_map are safe under the C++ standard.
            {
                const auto it = sub_HNSW_->label_lookup_.find(
                    static_cast<hnswlib::labeltype>(ext_lbl));
                if (it == sub_HNSW_->label_lookup_.end() ||
                        it->second != iid)
                    continue;
            }

            float* vec_ptr  = reinterpret_cast<float*>(sub_HNSW_->getDataByInternalId(iid));

            hnswlib::labeltype center = metaHNSW->searchKnn(vec_ptr, 1).top().second;
            int p = partitions[center] + 1;   // executor rank = shard index + 1

            if (p != static_cast<int>(node_id_)) {
                #pragma omp critical(delta_classify)
                {
                    send_vecs[p].insert(send_vecs[p].end(), vec_ptr, vec_ptr + dim_);
                    send_labels[p].push_back(ext_lbl);
                }
            }
        }
    }

    // Mark-delete pass: mark-delete for distinct labels is thread-safe under a
    // shared lock (same convention as markDeleteLocalBatch).
    {
        std::shared_lock<std::shared_mutex> lk(graph_mutex_);
        for (int p = 1; p < world_size; p++) {
            #pragma omp parallel for schedule(dynamic) num_threads(num_building_threads)
            for (int i = 0; i < static_cast<int>(send_labels[p].size()); i++) {
                sub_HNSW_->markDelete(static_cast<hnswlib::labeltype>(send_labels[p][i]));
            }
        }
    }

    size_t n_departing = 0;
    for (int p = 1; p < world_size; p++)
        n_departing += send_labels[p].size();

    std::vector<int> num_to_send(world_size, 0);
    for (int p = 1; p < world_size; p++)
        num_to_send[p] = static_cast<int>(send_labels[p].size());

    double end_iter = MPI_Wtime();
    std::cout << "[Executor " << node_id_ << "] delta rebuild iterate: departed="
              << n_departing << "\n";

    // ── Step 3: AllToAll exchange ─────────────────────────────────────────────
    // All ranks (including the coordinator with empty buffers) call the same
    // three collectives, so no sub-communicator is needed.
    //
    // Round 1 — MPI_Alltoall: exchange element counts so every rank knows how
    //           many elements it will receive from each peer.
    // Round 2 — MPI_Alltoallv on MPI_FLOAT: exchange the actual vectors.
    // Round 3 — MPI_Alltoallv on MPI_INT: exchange the corresponding labels.
    //           (Receive counts are derived from round 1; no extra Alltoall.)
    double start_exchange = MPI_Wtime();

    // Round 1: element counts (one int per rank).
    std::vector<int> recv_counts(world_size, 0);
    MPI_Alltoall(num_to_send.data(), 1, MPI_INT,
                 recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    const int total_recv = std::accumulate(recv_counts.begin(), recv_counts.end(), 0);

    // Round 2: vectors — convert element counts to float counts for Alltoallv.
    std::vector<int> send_vcounts(world_size), send_vdispls(world_size, 0);
    std::vector<int> recv_vcounts(world_size), recv_vdispls(world_size, 0);
    for (int p = 0; p < world_size; p++) {
        send_vcounts[p] = num_to_send[p] * static_cast<int>(dim_);
        recv_vcounts[p] = recv_counts[p]  * static_cast<int>(dim_);
    }
    for (int p = 1; p < world_size; p++) {
        send_vdispls[p] = send_vdispls[p-1] + send_vcounts[p-1];
        recv_vdispls[p] = recv_vdispls[p-1] + recv_vcounts[p-1];
    }

    std::vector<float> flat_send_vecs;
    { size_t tot = 0; for (auto& b : send_vecs) tot += b.size(); flat_send_vecs.reserve(tot); }
    for (int p = 0; p < world_size; p++)
        flat_send_vecs.insert(flat_send_vecs.end(), send_vecs[p].begin(), send_vecs[p].end());

    std::vector<float> flat_recv_vecs(static_cast<size_t>(total_recv) * dim_);
    MPI_Alltoallv(flat_send_vecs.data(), send_vcounts.data(), send_vdispls.data(), MPI_FLOAT,
                  flat_recv_vecs.data(), recv_vcounts.data(), recv_vdispls.data(), MPI_FLOAT,
                  MPI_COMM_WORLD);

    // Round 3: labels — reuse element counts directly (one int per element).
    std::vector<int> send_ldispls(world_size, 0), recv_ldispls(world_size, 0);
    for (int p = 1; p < world_size; p++) {
        send_ldispls[p] = send_ldispls[p-1] + num_to_send[p-1];
        recv_ldispls[p] = recv_ldispls[p-1] + recv_counts[p-1];
    }

    std::vector<int> flat_send_labels;
    { size_t tot = 0; for (auto& b : send_labels) tot += b.size(); flat_send_labels.reserve(tot); }
    for (int p = 0; p < world_size; p++)
        flat_send_labels.insert(flat_send_labels.end(), send_labels[p].begin(), send_labels[p].end());

    std::vector<int> flat_recv_labels(total_recv);
    MPI_Alltoallv(flat_send_labels.data(), num_to_send.data(), send_ldispls.data(), MPI_INT,
                  flat_recv_labels.data(), recv_counts.data(),  recv_ldispls.data(), MPI_INT,
                  MPI_COMM_WORLD);

    double end_exchange = MPI_Wtime();

    std::cout << "[Executor " << node_id_ << "] delta rebuild exchange: arrived="
              << total_recv << "\n";

    // ── Step 4: resize if needed, then insert with replace_deleted ────────────
    // After the mark-delete pass, sub_HNSW_->num_deleted_ counts every slot
    // currently available for reuse (including any prior stream-deleted elements
    // that were never replaced).  Incoming insertions with replace_deleted=true
    // fill those slots first; only the excess needs new capacity.
    {
        const size_t total_freed = sub_HNSW_->num_deleted_.load();
        if (static_cast<size_t>(total_recv) > total_freed) {
            const size_t net_new = static_cast<size_t>(total_recv) - total_freed;
            std::unique_lock<std::shared_mutex> lk(graph_mutex_);
            const size_t needed = sub_HNSW_->getCurrentElementCount() + net_new;
            if (needed > sub_HNSW_->getMaxElements())
                sub_HNSW_->resizeIndex(needed);
        }
    }

    double start_hnsw = MPI_Wtime();
    {
        std::shared_lock<std::shared_mutex> lk(graph_mutex_);
        #pragma omp parallel for schedule(dynamic) num_threads(num_building_threads)
        for (int i = 0; i < total_recv; i++) {
            sub_HNSW_->addPoint(
                flat_recv_vecs.data() + static_cast<size_t>(i) * dim_,
                static_cast<hnswlib::labeltype>(flat_recv_labels[i]),
                /*replace_deleted=*/true);
        }
    }
    double end_hnsw = MPI_Wtime();

    // ── Step 5: compute and log remaining unfilled deleted slots ──────────────
    // With allow_replace_deleted=true, deleted_elements is kept in sync with
    // num_deleted_: every markDelete adds to it, every successful slot-reuse by
    // addPoint removes from it.  So deleted_elements.size() == num_deleted_ and
    // both equal the number of tombstone slots still sitting in the graph.
    size_t remaining_deleted;
    {
        std::lock_guard<std::mutex> del_lock(sub_HNSW_->deleted_elements_lock);
        remaining_deleted = sub_HNSW_->deleted_elements.size();
    }
    // Sanity cross-check (both fields should agree when replace_deleted=true).
    assert(remaining_deleted == sub_HNSW_->num_deleted_.load() &&
           "deleted_elements.size() and num_deleted_ out of sync");

    std::cout << "[Executor " << node_id_ << "] delta rebuild complete:"
              << "  departed=" << n_departing
              << "  arrived=" << total_recv
              << "  remaining_deleted_slots=" << remaining_deleted
              << "\n";

    last_rebuild_remaining_deleted_ = remaining_deleted;
    last_rebuild_iterate_s_         = end_iter     - start_iter;
    last_rebuild_exchange_s_        = end_exchange - start_exchange;
    last_rebuild_graph_s_           = end_hnsw     - start_hnsw;

    delete metaHNSW;

    MessageHeader header;
    header.type = REBUILD_SUCCESS;
    header.size = sub_HNSW_->maxlevel_;
    MPI_Send(&header, sizeof(MessageHeader), MPI_BYTE, 0, REBUILD_SUCCESS,
             MPI_COMM_WORLD);
}

void Executor::reBuildReplace(
    int meta_size,
    int ncenters,
    int world_size,
    int ef_construction,
    int num_building_threads
) {
    std::cout << "[Executor " << node_id_ << "] rebuild (replace-deleted)\n";
    if (num_building_threads == -1) {
        num_building_threads = omp_get_max_threads();
    }

    std::vector<float*> active_vectors;
    std::vector<int> active_indices;

    {
        std::shared_lock lock(graph_mutex_);
        size_t nelts = sub_HNSW_->getCurrentElementCount();
        active_vectors.reserve(nelts);
        active_indices.reserve(nelts);
        for (hnswlib::tableint internal_id = 0; internal_id < nelts; internal_id++) {
            if (!sub_HNSW_->isMarkedDeleted(internal_id)) {
                float* vec_ptr = reinterpret_cast<float*>(sub_HNSW_->getDataByInternalId(internal_id));
                size_t external_label = sub_HNSW_->getExternalLabel(internal_id);
                active_vectors.push_back(vec_ptr);
                active_indices.push_back(static_cast<int>(external_label));
            }
        }
    }

    std::vector<char> buffer(meta_size);
    MPI_Recv(buffer.data(), meta_size, MPI_BYTE, 0, META_HNSW_SEND, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    std::ofstream outfile("tmp_hnsw_received.bin", std::ios::binary);
    outfile.write(buffer.data(), meta_size);
    outfile.close();

    hnswlib::HierarchicalNSW<float>* metaHNSW =
        new hnswlib::HierarchicalNSW<float>(space_, "tmp_hnsw_received.bin");

    std::vector<int> partitions(ncenters);
    MPI_Recv(partitions.data(), ncenters, MPI_INT, 0, META_PARTITIONS_SEND, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    std::vector<std::vector<float>> partition_vectors(world_size);
    std::vector<std::vector<int>> partition_indices(world_size);
    std::vector<int> labels_to_delete;

    #pragma omp parallel for num_threads(num_building_threads)
    for (int i = 0; i < (int)active_vectors.size(); i++) {
        float* vec = active_vectors[i];
        int index = active_indices[i];
        hnswlib::labeltype label = metaHNSW->searchKnn(vec, 1).top().second;
        int p = partitions[label] + 1;

        if (p != node_id_) {
            #pragma omp critical(rebuild_replace_partitions)
            {
                partition_vectors[p].insert(partition_vectors[p].end(), vec, vec + dim_);
                partition_indices[p].push_back(index);
                labels_to_delete.push_back(index);
            }
        }
    }

    std::vector<int> num_to_recv(world_size, 0);
    std::vector<int> num_to_send(world_size, 0);
    for (int i = 1; i < world_size; i++) {
        num_to_send[i] = partition_vectors[i].size() / dim_;
    }

    std::vector<MPI_Request> requests;
    for (int peer = 1; peer < world_size; ++peer) {
        if (peer == node_id_) continue;
        requests.emplace_back();
        MPI_Irecv(&num_to_recv[peer], 1, MPI_INT, peer, TAG_SEND_NUM, MPI_COMM_WORLD, &requests.back());
        requests.emplace_back();
        MPI_Isend(&num_to_send[peer], 1, MPI_INT, peer, TAG_SEND_NUM, MPI_COMM_WORLD, &requests.back());
    }
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    requests.clear();

    int total_recv = std::accumulate(num_to_recv.begin(), num_to_recv.end(), 0);
    std::vector<float> all_recv_vectors(total_recv * dim_);
    std::vector<int> all_recv_labels(total_recv);

    std::vector<int> offset(world_size, 0);
    for (int i = 1; i < world_size; ++i) {
        offset[i] = offset[i - 1] + num_to_recv[i - 1];
    }

    for (int peer = 1; peer < world_size; ++peer) {
        if (peer == node_id_) continue;

        if (num_to_recv[peer] > 0) {
            int n_recv = num_to_recv[peer];
            int peer_offset = offset[peer];
            requests.emplace_back();
            MPI_Irecv(all_recv_vectors.data() + peer_offset * dim_, n_recv * dim_, MPI_FLOAT, peer, TAG_SEND_VECS, MPI_COMM_WORLD, &requests.back());
            requests.emplace_back();
            MPI_Irecv(all_recv_labels.data() + peer_offset, n_recv, MPI_INT, peer, TAG_SEND_LABELS, MPI_COMM_WORLD, &requests.back());
        }

        if (num_to_send[peer] > 0) {
            requests.emplace_back();
            MPI_Isend(partition_vectors[peer].data(), num_to_send[peer] * dim_, MPI_FLOAT, peer, TAG_SEND_VECS, MPI_COMM_WORLD, &requests.back());
            requests.emplace_back();
            MPI_Isend(partition_indices[peer].data(), num_to_send[peer], MPI_INT, peer, TAG_SEND_LABELS, MPI_COMM_WORLD, &requests.back());
        }
    }
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

    {
        std::unique_lock lock(graph_mutex_);

        size_t required = sub_HNSW_->getCurrentElementCount() + total_recv;
        if (required > sub_HNSW_->getMaxElements()) {
            sub_HNSW_->resizeIndex(required);
        }

        for (int label : labels_to_delete) {
            try {
                sub_HNSW_->markDelete(label);
            } catch (...) {
                // Best-effort: label may already be deleted or absent.
            }
        }

        #pragma omp parallel for num_threads(num_building_threads)
        for (int i = 0; i < total_recv; i++) {
            float* vec = all_recv_vectors.data() + (i * dim_);
            int index = all_recv_labels[i];
            sub_HNSW_->addPoint(vec, index, true);
        }

        sub_HNSW_->setEf(ef_construction);
    }

    delete metaHNSW;

    MessageHeader header;
    header.type = REBUILD_SUCCESS;
    MPI_Send(&header, sizeof(MessageHeader), MPI_BYTE, 0, REBUILD_SUCCESS, MPI_COMM_WORLD);
}

void Executor::search(size_t k, int tag) {
    float query_vector[dim_];
    comm_.recv_query(query_vector, dim_, tag);
    int results[k];

    std::shared_lock lock(graph_mutex_);
    std::vector<std::pair<float, hnswlib::labeltype>> result = sub_HNSW_->searchKnnCloserFirst(query_vector, k);

    int i = 0;
    for (std::pair<float, hnswlib::labeltype> ref : result) {
        // float* elt = (float*) sub_HNSW_->getDataByInternalId(ref.second);
        // std::copy(elt, elt + dim, results + (i * dim));
        results[i] = ref.second;
        i++;
    }

    comm_.send_result(results, k, tag);
}

void Executor::batch_search(size_t num_queries, size_t k, int tag) {
    float* query_vectors = new float[num_queries * dim_];

    comm_.recv_vector_batch(query_vectors, num_queries, dim_, 0, tag);
    int* results = new int[num_queries * k];

    std::shared_lock lock(graph_mutex_);
    #pragma omp parallel for
    for (int i = 0; i < (int)num_queries; i++) {
        float* query_vector = query_vectors + (i * dim_);
        std::vector<std::pair<float, hnswlib::labeltype>> result = sub_HNSW_->searchKnnCloserFirst(query_vector, k);
        for (int j = 0; j < (int)result.size(); j++) {
            results[i * k + j] = result[j].second;
        }
    }

    // void send_result_batch(const int* results, size_t num_queries, size_t num_neighbors, int tag)
    comm_.send_result_batch(results, num_queries, k, tag);

    delete[] query_vectors;
    delete[] results;

}

size_t Executor::getElementCount() const {
    std::shared_lock lock(graph_mutex_);
    // cur_element_count includes soft-deleted entries; subtract them for the
    // live count.  Both accessors are thread-safe atomics in hnswlib.
    size_t total   = sub_HNSW_->getCurrentElementCount();
    size_t deleted = sub_HNSW_->getDeletedCount();
    return (total > deleted) ? (total - deleted) : 0;
}

double Executor::getTombstoneRatio() const {
    std::shared_lock lock(graph_mutex_);
    const size_t total   = sub_HNSW_->getCurrentElementCount();
    if (total == 0) return 0.0;
    const size_t deleted = sub_HNSW_->getDeletedCount();
    return static_cast<double>(deleted) / static_cast<double>(total);
}

// ── insertLocalBatch ─────────────────────────────────────────────────────────
// Insert a batch of vectors (received via AllToAllV in shared_batch_experiment)
// directly into the local sub-HNSW without going through the MPI message
// protocol.  Resizes the index if capacity would be exceeded.
void Executor::insertLocalBatch(const std::vector<float>& vecs,
                                const std::vector<int>&   labels) {
    if (labels.empty()) return;
    const size_t incoming = labels.size();

    // Phase 1: capacity check and resize under exclusive lock.
    //
    // With replace_deleted=true (used in Phase 2), each addPoint reuses a
    // tombstone slot before claiming a fresh one.  The number of new internal
    // IDs actually needed is therefore:
    //
    //   net_new = incoming - min(num_deleted_, incoming)
    //
    // getCurrentElementCount() is the high-water mark of ever-allocated
    // internal IDs (live + tombstones, but NOT counting reused slots twice),
    // so the true needed capacity is current + net_new.
    // num_deleted_ is atomic so it is safe to read under the exclusive lock.
    {
        std::unique_lock<std::shared_mutex> lk(graph_mutex_);
        const size_t current  = sub_HNSW_->getCurrentElementCount();
        const size_t deleted  = sub_HNSW_->num_deleted_.load();
        const size_t reusable = std::min(deleted, incoming);
        const size_t net_new  = incoming - reusable;
        const size_t needed   = current + net_new;
        if (needed > sub_HNSW_->getMaxElements()) {
            const size_t new_max = std::max(
                needed,
                sub_HNSW_->getMaxElements() + sub_HNSW_->getMaxElements() / 2);
            sub_HNSW_->resizeIndex(new_max);
        }
    }

    // Phase 2: parallel inserts under shared lock.
    // replace_deleted=true reuses tombstone slots from prior stream-deletes
    // before allocating new capacity.  hnswlib addPoint is internally
    // thread-safe (per-element locks and deleted_elements_lock for slot
    // selection), so concurrent calls on distinct labels are safe here.
    std::shared_lock<std::shared_mutex> lk(graph_mutex_);
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < incoming; i++)
        sub_HNSW_->addPoint(vecs.data() + i * dim_, labels[i], /*replace_deleted=*/true);
}

// ── markDeleteLocal ──────────────────────────────────────────────────────────
// Mark a single vector deleted without MPI.  Ignores labels not present in
// this shard (the label may have been routed to a different executor).
void Executor::markDeleteLocal(int label) {
    std::unique_lock<std::shared_mutex> lk(graph_mutex_);
    try {
        sub_HNSW_->markDelete(static_cast<hnswlib::labeltype>(label));
    } catch (...) {
        // Label not found in this shard – silently skip.
    }
}

// ── markDeleteLocalBatch ─────────────────────────────────────────────────────
// Mark a batch of vectors deleted in parallel.  hnswlib's markDelete is
// internally thread-safe for distinct labels — it uses per-label
// label_op_locks_, a separate label_lookup_lock for the lookup table, and
// atomic num_deleted_ — so concurrent calls on different labels are safe under
// a shared (reader) lock on graph_mutex_.  Labels absent from this shard are
// silently skipped.
void Executor::markDeleteLocalBatch(const std::vector<int>& labels) {
    if (labels.empty()) return;
    std::shared_lock<std::shared_mutex> lk(graph_mutex_);
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < labels.size(); i++) {
        try {
            sub_HNSW_->markDelete(static_cast<hnswlib::labeltype>(labels[i]));
        } catch (...) {
            // Label not found in this shard – silently skip.
        }
    }
}

// ── searchLocalBatch ─────────────────────────────────────────────────────────
// Search a batch of query vectors (received via AllToAllV phase-1) and return
// results without MPI.  Each result vector is sorted nearest-first.
std::vector<std::vector<std::pair<float, hnswlib::labeltype>>>
Executor::searchLocalBatch(const std::vector<float>& queries,
                            size_t                     num_queries,
                            size_t                     k) {
    std::vector<std::vector<std::pair<float, hnswlib::labeltype>>> results(num_queries);
    if (num_queries == 0 || sub_HNSW_->getCurrentElementCount() == 0)
        return results;

    const size_t k_eff = std::min(k, sub_HNSW_->getCurrentElementCount());

    std::shared_lock<std::shared_mutex> lk(graph_mutex_);

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_queries; i++) {
        const float* q = queries.data() + i * dim_;
        auto pq = sub_HNSW_->searchKnn(q, k_eff);
        auto& res = results[i];
        res.reserve(pq.size());
        // searchKnn returns a max-heap (farthest on top); pop to get all, then
        // reverse so the result vector is sorted nearest-first.
        while (!pq.empty()) { res.push_back(pq.top()); pq.pop(); }
        std::reverse(res.begin(), res.end());
    }
    return results;
}

void Executor::insert_batch(size_t num_vecs, int tag) {
    // Receive the vector data and labels sent by Coordinator::handle_inserts.
    // The coordinator sends: num_vecs*dim floats (tag), then num_vecs ints (tag).
    std::vector<float> vecs(num_vecs * dim_);
    MPI_Recv(vecs.data(), num_vecs * dim_, MPI_FLOAT, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    std::vector<int> labels(num_vecs);
    MPI_Recv(labels.data(), num_vecs, MPI_INT, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Resize if needed, then insert all vectors.
    {
        std::unique_lock write_lock(graph_mutex_);
        size_t current = sub_HNSW_->getCurrentElementCount();
        if (current + num_vecs > sub_HNSW_->getMaxElements()) {
            sub_HNSW_->resizeIndex((current + num_vecs) * 2);
        }
        for (size_t i = 0; i < num_vecs; i++) {
            sub_HNSW_->addPoint(vecs.data() + i * dim_, labels[i], /*replace_deleted=*/true);
        }
    }

    comm_.send_ack(INSERT_BATCH_SUCCESS, 0, tag);
}

void Executor::insert(int tag) {
    float insert_vector[dim_];
    int label;
    comm_.recv_insert(insert_vector, dim_, label, tag);
    // First take shared lock and attempt to add if there's capacity
    {
        std::shared_lock read_lock(graph_mutex_);
        // +1 because we're about to add one element
        if (sub_HNSW_->getCurrentElementCount() + 100 <= sub_HNSW_->getMaxElements()) {
            sub_HNSW_->addPoint(insert_vector, label, /*replace_deleted=*/true);

            comm_.send_ack(INSERT_SUCCESS, 0, tag);
            return;
        }
    }
    // otherwise drop shared lock and take writer lock to resize
    {
        std::unique_lock write_lock(graph_mutex_);
        // re-check condition (another thread may have resized already)
        if (sub_HNSW_->getCurrentElementCount() == sub_HNSW_->getMaxElements()) {
            sub_HNSW_->resizeIndex(sub_HNSW_->getMaxElements() * 2);
        }
    }

    // After resize, safe to add. Acquire shared lock again
    {
        std::shared_lock read_lock(graph_mutex_);
        sub_HNSW_->addPoint(insert_vector, label, /*replace_deleted=*/true);
    }
    comm_.send_ack(INSERT_SUCCESS, 0, tag);
}

void Executor::delete_vector(size_t label, int tag) {
    std::shared_lock lock(graph_mutex_);
    std::vector<float> data;
    bool found = false;
    try {
        data = sub_HNSW_->getDataByLabel<float>(label);
        found = true;
    } 
    catch (const std::runtime_error& error ) {
        found = false;
    }

    MessageHeader ack;
    if (found) {
        sub_HNSW_->markDelete(label);
        comm_.send_ack(DELETE_SUCCESS, 0, tag);
        comm_.send_vector(data.data(), dim_, 0, tag);
    } else {
        // tell coordinator you didn't have it
        comm_.send_ack(NOT_FOUND, 0, tag);
    }
}

std::string Executor::save(const std::string& prefix) {
    std::shared_lock lock(graph_mutex_);
    char suffix[10];
    snprintf(suffix, 10, "_%lu.bin", node_id_);
    std::string hnsw_path = prefix + suffix;
    std::cout << "[Executor " << node_id_ << " ] Saving sub-HNSW to: " << hnsw_path << "\n";
    sub_HNSW_->saveIndex(hnsw_path);
    std::cout << "[Executor " << node_id_ << " ] Saved\n";
    return hnsw_path;
}

void Executor::load(const std::string& prefix, int ef_search) {
    char suffix[10];
    snprintf(suffix, 10, "_%lu.bin", node_id_);
    std::string hnsw_path = prefix + suffix;

    std::cout << "[Executor " << node_id_ << " ] Loading sub-HNSW from: " << hnsw_path << "\n";
    sub_HNSW_ = new hnswlib::HierarchicalNSW<float>(space_, hnsw_path, false, 0, true);
    sub_HNSW_->setEf(ef_search);
}


// for running MIPS
Point transformVector(float* vec, size_t dim, float alpha) {
    Point transformed; transformed.resize(dim + 1);
    float dot = 0.0f;
    for (int i = 0; i < dim; i++) {
        dot += vec[i] * vec[i];
        transformed[i] = vec[i];
    }
    float norm = sqrt(dot);
    transformed[dim] = (alpha * norm);

    return transformed;
}