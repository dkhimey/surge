#include <iostream>
#include <mpi.h>
#include <omp.h>
#include "index.h"
#include <yaml-cpp/yaml.h>

std::unordered_map<std::string, int> PARAMS = {
    {"num_neighbors_request", 10},
    {"branching_factor", 10},
    {"recall_target", 0.9},
    {"nprobes", 5}
};

int main(int argc, char **argv) {
    int node, world_size;

    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <dataset> <num_partitions> <maintenance_threshold>\n";
        return 1;
    }

    std::string dataset_name = argv[1];
    int num_partitions = std::stoi(argv[2]);
    int maintenance_threshold = std::stoi(argv[3]);

    std::string log_id = "streaming_" + dataset_name + "_" + std::to_string(num_partitions);
    Log logger(log_id);

    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    if (provided < MPI_THREAD_MULTIPLE) {
        std::cerr << "Error: MPI does not provide required threading level (MPI_THREAD_MULTIPLE)\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &node);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    Communicator comm;

    if (world_size == 1) {
        // TODO
        std::cerr << "Not Implemented: Single Node SURGE";
        return 1;
    }

    if (world_size != num_partitions + 1) {
        std::cerr << "ERROR: number of processes (" << world_size << ") should be one more than the number of partitions (" << num_partitions << ")\n";
        return 1;
    }

    std::pair<int,int> data_info = get_dataset_info(DATASETS[dataset_name]["base_file"]);
    int nvectors = data_info.first;
    int dim = data_info.second;

    if (node == 0) { // Coordinator
        // load fresh routing layer after each batch
        // check how many move partitions

        std::pair<int, int> data_info = get_dataset_info(DATASETS[dataset_name]["base_file"]);
        int nvectors = data_info.first;
        int dim = data_info.second;
        // std::vector<float> queries = readVecs(DATASETS[dataset_name]["base_file"], config.benchmark_params.format, dim);
        
        YAML::Node root = YAML::LoadFile(DATASETS[dataset_name]["runbook"]);
        // "/dataset/big-ann-benchmarks/final_runbook_inserts.yaml"
        std::cout << "[Coordinator] loaded runbook file\n";
        // Access top-level key
        YAML::Node steps = root[dataset_name];
        if (!steps) {
            std::cerr << "Key '" << dataset_name << "' not found!\n";
            return 1;
        }

        std::string streaming_log_path = logger.log_dir +  "/rebuild_" 
            + std::to_string(maintenance_threshold)
            + "_threshold.txt";

        std::ofstream outFile(streaming_log_path, std::ios::out | std::ios::app);

        
    } else {

    }
}