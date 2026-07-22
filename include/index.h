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

    ~Coordinator();

    // Owns raw pointers (space_, meta_HNSW_, cached rebuild graph); non-copyable
    // to avoid double-free. Only ever constructed in place / on the heap.
    Coordinator(const Coordinator&)            = delete;
    Coordinator& operator=(const Coordinator&) = delete;

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

    // Load coordinator state from cluster analysis output
    // Reads step_<step>_hnsw.bin, _centers.csv, _center_counts.csv, _labels.csv,
    // and partitions file (center -> shard mapping)
    void loadFromClusterAnalysis(
        const std::string& state_dir,
        int                step,
        const std::string& partitions_file,
        int                num_partitions,
        int                ef_search,
        int                init_start);

    // Expose routing state for replication across ranks
    hnswlib::HierarchicalNSW<float>* getMetaHNSW() { return meta_HNSW_; }
    const std::vector<int>& getPartitions() const { return partitions; }
    const std::unordered_map<int,int>& getLabelToCenter() const { return label_to_center_; }
    const std::vector<int>& getCenterCounts() const { return center_counts_; }

    // Batch updates with two-phase locking: accumulate in parallel, apply under write lock
    void updateCentersForInsertBatch(const std::vector<float>& vecs,
                                      const std::vector<int>& center_ids);

    void updateCentersForDeleteBatch(const std::vector<float>& vecs, 
                                     const std::vector<int>& closest_center_labels);
    // Check if rebuild needed without executing; caches result (returns 0/1/2)
    int checkNeedRebuild(int full_threshold, int partial_threshold,
                         int ef_construction, int M_meta);

    // Execute cached rebuild from checkNeedRebuild; serial only
    void doRebuildSimple(int world_size);

    // Delta rebuild: mark-delete departing elements, insert arriving without reconstructing
    void doRebuildDelta(int world_size);

    // Force full rebuild independent of thresholds; used for external conditions like tombstone ratio
    void doForceFullRebuild(int world_size, int ef_construction, int M_meta);

    // Returns statistics from most recent checkNeedRebuild (all zero before first call)
    int    getCachedElementsMoved()  const { return cached_elements_moved_; }
    int    getCachedCentersMoved()   const { return cached_centers_moved_; }
    double getCachedRepartHnswS()    const { return cached_repart_hnsw_s_; }
    double getCachedRepartBottomS()  const { return cached_repart_bottom_s_; }
    double getCachedRepartKaffpaS()  const { return cached_repart_kaffpa_s_; }
    double getCachedRepartRelabelS() const { return cached_repart_relabel_s_; }

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
        std::vector<int>* preassigned_partitions = nullptr,
        int start_offset = 0
    );

    int rePartition(
        std::vector<int>& new_partitions,
        hnswlib::HierarchicalNSW<float>*& new_meta_HNSW,
        int ef_construction,
        int M_meta,
        double* out_hnsw_s    = nullptr,  // time to build new meta-HNSW
        double* out_bottom_s  = nullptr,  // time to extract bottom layer
        double* out_kaffpa_s  = nullptr,  // time to run kaffpa
        double* out_relabel_s = nullptr   // time to relabel partitions
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
    hnswlib::HierarchicalNSW<float>* meta_HNSW_ = nullptr;

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
    int                              cached_rebuild_type_    = 0;   // 0=none,1=full,2=partial
    int                              cached_elements_moved_  = 0;   // sum of center_counts_ for moved centers
    int                              cached_centers_moved_   = 0;   // number of centers that changed shard
    double                           cached_repart_hnsw_s_   = 0.0; // time to build new meta-HNSW
    double                           cached_repart_bottom_s_ = 0.0; // time to extract bottom layer
    double                           cached_repart_kaffpa_s_ = 0.0; // time to run kaffpa
    double                           cached_repart_relabel_s_= 0.0; // time for partition relabeling

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

    ~Executor();

    // Owns raw pointers (space_, sub_HNSW_); non-copyable to avoid double-free.
    Executor(const Executor&)            = delete;
    Executor& operator=(const Executor&) = delete;

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

    // Direct (MPI-free) batch operations for distributed experiments
    // Insert vectors received via AllToAllV; exclusive lock for full batch
    void insertLocalBatch(const std::vector<float>& vecs,
                          const std::vector<int>&   labels);

    // Mark single vector deleted; silently ignores missing labels
    void markDeleteLocal(int label);

    // Batch delete with shared lock (thread-safe per-element in hnswlib)
    void markDeleteLocalBatch(const std::vector<int>& labels);

    // Search batch of queries; returns sorted results (nearest-first)
    std::vector<std::vector<std::pair<float, hnswlib::labeltype>>>
    searchLocalBatch(const std::vector<float>& queries,
                     size_t                     num_queries,
                     size_t                     k);

    // Return active (non-deleted) elements in sub-HNSW
    size_t getElementCount() const;

    // Return fraction of deleted slots: num_deleted / getCurrentElementCount
    double getTombstoneRatio() const;

    // Per-phase timing from most recent rebuild (zero before first call)
    double getLastRebuildIterateS()  const { return last_rebuild_iterate_s_; }
    double getLastRebuildExchangeS() const { return last_rebuild_exchange_s_; }
    double getLastRebuildGraphS()    const { return last_rebuild_graph_s_; }

    // Count of deleted slots not yet reused after most recent delta rebuild
    size_t getLastRebuildRemainingDeleted() const { return last_rebuild_remaining_deleted_; }

    // Labels of vectors that arrived during most recent rebuild (for label_to_shard sync)
    const std::vector<int>& getLastRebuildMovedLabels() const { return last_rebuild_moved_labels_; }

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

    // Delta rebuild: mark-delete migrated-out elements, insert migrated-in with replace_deleted=true
    // Graph topology retained; not reconstructed from scratch
    void reBuildDelta(
        int meta_size,
        int ncenters,
        int world_size,
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

    hnswlib::HierarchicalNSW<float>* sub_HNSW_ = nullptr;

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

    // Per-phase timing from most recent rebuild
    double last_rebuild_iterate_s_  = 0.0; // route existing elements
    double last_rebuild_exchange_s_ = 0.0; // MPI exchange
    double last_rebuild_graph_s_    = 0.0; // insert into new sub-HNSW

    // Deleted slots unreplaced after most recent delta rebuild
    // Equals sub_HNSW_->deleted_elements.size() after insertions complete
    size_t last_rebuild_remaining_deleted_ = 0;

    // Arrived labels for label_to_shard sync via Allgatherv after rebuild
    std::vector<int> last_rebuild_moved_labels_;
};