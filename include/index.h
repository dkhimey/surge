#pragma once

#include "communicator.h"
#include "kaHIP_interface.h"
#include <shared_mutex>
#include <unordered_map>

#define KMEANS_EPOCHS 100
#define VECTOR_BATCH_SIZE 100000

enum class RoutingMode {
    BranchingFactor, // param = branching factor (number of nearest centroids to search)
    NProbe,          // param = number of unique partitions to collect
    RecallTarget,    // param = target recall in [0, 1]
};

// COORDINATOR
class Coordinator {
public:
    Coordinator(
        int dim,
        Log* logger
    );

    Coordinator(
        int dim,
        Communicator* comm,
        Log* logger
    );

    void setSampleData(float* data, size_t count);
    void build(
        int ncenters,
        int num_partitions,
        int ef_construction,
        int M_meta  
    );
    
    // Coordinator-specific helper
    void load(const std::string& dir_path, int ef_search);
    void save(const std::string& output_dir);

    // ── Accessors for shared_batch_experiment ──────────────────────────────
    // Expose routing state so the experiment can replicate it on all ranks.
    hnswlib::HierarchicalNSW<float>* getMetaHNSW() { return meta_HNSW_; }
    const std::vector<int>& getPartitions() const { return partitions; }
    const std::unordered_map<int,int>& getLabelToCenter() const { return label_to_center_; }

    // Two-phase, lock-minimal center updates for batch insert/delete steps.
    // Phase 1 accumulates per-center sums in parallel; Phase 2 applies all
    // updates under a single write lock.  Called by coordinator after the
    // Allgatherv that populates label_to_center on all ranks.
    void updateCentersForInsertBatch(const std::vector<float>& vecs,
                                      const std::vector<int>& center_ids);

    void updateCentersForDeleteBatch(const std::vector<float>& vecs, 
                                     const std::vector<int>& closest_center_labels);
    // Check whether a rebuild is needed without executing it.  Runs
    // rePartition, caches the result (new meta-HNSW, new partitions,
    // serialised buffer).  Returns 0 = no rebuild, 1 = full, 2 = partial.
    // Must be followed by doRebuildSimple() if the return value is non-zero.
    int checkNeedRebuild(int full_threshold, int partial_threshold,
                         int ef_construction, int M_meta);

    // Execute the cached rebuild prepared by checkNeedRebuild().  Sends
    // FULL/PARTIAL_REBUILD_REQUEST + meta-HNSW + partitions to every
    // executor, waits for REBUILD_SUCCESS from each, then swaps the new
    // meta-HNSW and partitions into the Coordinator's internal state.
    // Safe to call only in serial (non-concurrent) rebuild scenarios.
    void doRebuildSimple(int world_size, bool incremental = false);

    void load_gp(const std::string& prefix, int ef_search);
    void setEfSearch(int ef_search);

    // std::vector<size_t> routeQuery(float* query, size_t top_n = 1) const;
    // std::vector<size_t> getPartition(float* vec, int n = 1);
    // std::vector<size_t> getPartition(float* vec, int n = 1, float* dist = nullptr);
    
    std::vector<int> distribute_vectors(
        const std::string& base_file, 
        int total_vectors, 
        bool log_partitions,
        int num_threads = -1,
        std::vector<int>* preassigned_partitions = nullptr
    );

    int rePartition(
        std::vector<int>& new_partitions, 
        hnswlib::HierarchicalNSW<float>*& new_meta_HNSW, 
        int ef_construction, 
        int M_meta
    );

    int reBuild(int world_size, int ef_construction, int M_meta, int full_threshold = 0, int partial_threshold = 0);

    std::vector<size_t> route_query(
        float* query_vector,
        size_t branching_factor,
        float* dist = nullptr
    );

    std::vector<size_t> route_query_nprobe(
        float* query_vector,
        size_t nprobe,
        float* dist = nullptr
    );

    std::vector<size_t> route_query_recall_target(
        float* query_vector,
        float recall_target,
        float* dist = nullptr
    );

    std::vector<std::vector<size_t>> route_queries(
        const std::vector<float>& query_vectors,
        RoutingMode mode,
        float param
    );

    std::vector<std::vector<int>> handle_queries(
        const std::vector<float>& query_vectors,
        size_t num_neighbors,
        size_t branching_factor
    );

    // void handle_query(
    //     float* query_vector,
    //     int query_idx,
    //     const std::vector<int>& ground_truth_idx,
    //     size_t num_neighbors,
    //     size_t branching_factor,
    //     size_t num_closest_from_ground,
    //     std::vector<double>& latencies,
    //     std::atomic<size_t>& correct,
    //     std::atomic<size_t>& completed,
    //     std::vector<int>* access_rate = nullptr,
    //     std::vector<std::atomic<int>>* executor_hits = nullptr,
    //     float* recall = nullptr
    // );

    // std::vector<int> handle_query(
    //     float* query_vector,
    //     int query_idx,
    //     size_t num_neighbors,
    //     size_t branching_factor,
    //     std::vector<std::atomic<int>>* executor_hits = nullptr
    // );

    std::vector<int> handle_query(
        float* query_vector,
        int query_idx,
        size_t num_neighbors,
        RoutingMode mode,
        float param,
        std::vector<std::atomic<int>>* executor_hits = nullptr
    );

    void handle_insert(
        float* insert_vector,
        int label,
        int insert_idx,
        std::vector<float>& insert_distances,
        std::atomic<size_t>& completed
    );

    void handle_inserts(
        std::vector<float>& insert_vectors, 
        std::vector<int>& labels
    );

    bool handle_delete(int label, int world_size);

    void handle_deletes(const std::vector<int>& labels, int world_size);
    
    size_t getCurrentPartition(float* vec);
    
private:
    Log* logger_;
    Communicator* comm_ = nullptr;

    size_t dim_;
    size_t num_partitions_;
    size_t ncenters_;

    float* sample_data_;
    size_t sample_count_;

    hnswlib::SpaceInterface<float>* space_;
    std::function<float(float*, float*)> computeDistance_;
    hnswlib::HierarchicalNSW<float>* meta_HNSW_;

    std::mt19937 gen_;

    std::vector<int> center_counts_;
    std::vector<float> centers_;
    std::vector<int> partitions;
    std::atomic<RebuildState> rebuild_state = IDLE;

    // building helpers
    int kmeans_(float* sample, size_t nPrime, size_t m_centers, float* centers, float EPSILON = 1e-4f);
    std::pair<std::vector<int>, std::vector<int>> getBottomLayer_(hnswlib::HierarchicalNSW<float>* graph = nullptr);
    std::pair<int, std::vector<int>> matchPartitions_(const std::vector<int>& part1, const std::vector<int>& part2);

    // query helpers
    std::vector<size_t> getPartitionsForSearch_Branching_(float* vec, int branching_factor, float* dist = nullptr);
    std::vector<size_t> getPartitionsForSearch_Nprobe_(float* vec, int nprobe, float* dist = nullptr);
    std::vector<size_t> getPartitionsForSearch_RecallTgt_(float* vec, float recall_target, float* dist = nullptr);
    std::vector<size_t> getPartitionsForInsert_(float* vec, size_t label, float* dist = nullptr);
    std::vector<size_t> getPartitionsForDelete_(size_t label);
    std::vector<std::pair<float, hnswlib::labeltype>> findClosestCenters_(float* vec, size_t n);
    std::vector<size_t> convertCentersToPartitions_(const std::vector<std::pair<float, hnswlib::labeltype>>& centers, float* dist);
    void updateCentersForInsert_(float* vec, size_t label, size_t closest_center_label);
    void updateCentersForDelete_(std::vector<float>& vec, int closest_center_label);

    // Cached materials for the checkNeedRebuild / doRebuildSimple split.
    hnswlib::HierarchicalNSW<float>* cached_new_meta_HNSW_  = nullptr;
    std::vector<int>                 cached_new_partitions_;
    std::vector<char>                cached_hnsw_buffer_;
    int                              cached_rebuild_type_    = 0; // 0=none,1=full,2=partial

    // during insert, check if rebuild is already in progress, if so, just add to rebuild log
    std::atomic<bool> rebuild_pending_ = false; // indicates a rebuild must occur after the current one
    
    std::mutex insert_log_mutex_; // to insert into the rebuild log
    std::mutex delete_log_mutex_;
    std::shared_mutex center_mutex_; // to update centers
    std::shared_mutex graph_mutex_; // to perform the HNSW swap

    std::vector<std::pair<float*, int>> insert_log_; // holds inserts that occur while a rebuild is in prgress
    std::vector<int> delete_log_; // holds deletes that occur while rebuild is in progress

    // to track where to delete
    std::unordered_map<int, int> label_to_center_;
};

// EXECUTOR
class Executor {
public:
    Executor(int node_id, int dim, Communicator& comm, Log* logger = nullptr);

    void receiveData(size_t nrecv_vecs);
    void setData(float* data, int* indices, size_t count);
    void build(
        int ef_construction,
        int M_sub,
        int num_building_threads = -1
    );

    void load(const std::string& prefix, int ef_search);
    std::string save(const std::string& prefix);
    void setEfSearch(int ef_search);

    void search(size_t k, int tag);
    void insert(int tag);
    void insert_batch(size_t num_vecs, int tag);
    void delete_vector(size_t label, int tag);

    // ── Direct (MPI-free) operations for shared_batch_experiment ──────────
    // Insert a batch of vectors received via AllToAllV without going through
    // the MPI message protocol.  Acquires an exclusive lock for the full batch.
    void insertLocalBatch(const std::vector<float>& vecs,
                          const std::vector<int>&   labels);

    // Mark a single vector as deleted.  Silently ignores labels not present
    // in this shard.  Uses an exclusive lock.
    void markDeleteLocal(int label);

    // Mark a batch of vectors deleted in parallel.  hnswlib markDelete is
    // internally thread-safe for distinct labels (per-element label_op_locks_,
    // atomic num_deleted_, deleted_elements_lock), so only a shared lock is
    // needed here.  Labels not present in this shard are silently skipped.
    void markDeleteLocalBatch(const std::vector<int>& labels);

    // Search a batch of query vectors received via AllToAllV.  Returns one
    // result vector per query; each result is sorted nearest-first.
    std::vector<std::vector<std::pair<float, hnswlib::labeltype>>>
    searchLocalBatch(const std::vector<float>& queries,
                     size_t                     num_queries,
                     size_t                     k);

    // Returns the number of active (non-deleted) elements in the sub-HNSW.
    size_t getElementCount() const;

    void batch_search(size_t num_queries, size_t k, int tag);

    void reBuild(
        int meta_size, 
        int ncenters, 
        int world_size, 
        int ef_construction, 
        int M_sub,
        int num_building_threads = -1
    );

    void reBuildReplace(
        int meta_size,
        int ncenters,
        int world_size,
        int ef_construction,
        int num_building_threads = -1
    );
    
    void partialReBuild(
        int meta_size, 
        int ncenters, 
        int world_size,
        int ef_construction,
        int num_building_threads = -1
    );
    
private:
    Communicator& comm_;

    hnswlib::HierarchicalNSW<float>* sub_HNSW_;

    mutable std::shared_mutex graph_mutex_;
    // when inserting, each thread holds the shared lock, when swapping hnsw in final step, get exclusive lock
    Log* logger_;
    size_t node_id_;

    size_t dim_;

    hnswlib::SpaceInterface<float>* space_;
    std::function<float(float*, float*)> computeDistance_;

    std::vector<float> local_vectors_;
    std::vector<int> local_indices_;

    float* data_;
    int* indices_;
    size_t data_count_ = 0;
};