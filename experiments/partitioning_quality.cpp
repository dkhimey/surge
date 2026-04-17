#include <iostream>
#include <mpi.h>
#include <omp.h>
#include "index.h"

int main(int argc, char **argv) {
    int node, world_size;

    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <dataset> <num_partitions>\n";
        return 1;
    }

    std::string dataset_name = argv[1];
    int num_partitions = std::stoi(argv[2]);

    std::string log_id = "partition_quality_" + dataset_name + "_" + std::to_string(num_partitions);
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

    printf("Max threads: %d\n", omp_get_max_threads());

    if (node == 0) { // Coordinator
        // take sample (just read in first sample_size elements, assume they are randomly distributed across file)

        size_t sample_size = 100000; //TODO: hard coded

        std::vector<float> sample = getSample(DATASETS[dataset_name]["base_file"], 
                                              nvectors, dim, sample_size);
        
        // build meta hnsw on the centers
        Coordinator metaIndex(dim, &comm, &logger);
        metaIndex.setSampleData(sample.data(), sample_size);

        int ncenters = 10000; //TODO: hard coded
        int ef_construction = 200; //TODO: hard coded
        int M_meta = 16; //TODO: hard coded

        metaIndex.build(ncenters, num_partitions, ef_construction, M_meta);

        // assign vectors to workers
        double start = MPI_Wtime();
        int num_threads = 32; //TODO: hard coded
        bool log_partitions = true; //TODO: hard coded
        metaIndex.distribute_vectors(DATASETS[dataset_name]["base_file"], nvectors, log_partitions, num_threads);
        double end = MPI_Wtime();

        logger.partition_time = end - start;
        std::cout << "Partition time: " << end - start << " seconds\n";

        // complete build process from Coordinator side
        comm.broadcast_termination(world_size);

        std::string meta_dir = dataset_name + "_" + std::to_string(num_partitions);
        metaIndex.save(meta_dir);
        logger.meta_index_path = meta_dir + "/metaHNSW.bin";
        logger.saveControllerLog();
    } else {
        std::cout << "[Executor " << node << "] log_id: " << log_id << "\n";
        Executor subIndex(node, dim, comm, &logger);

        size_t num_recv = 0;
        bool done = false;

        while (!done) {
            // recieve header
            MessageHeader recv_header;
            comm.recv_header(recv_header, 0);
            if (recv_header.type == END_OF_COMMUNICATION) {
                done = true;
            } else {
                assert(recv_header.type == VECTOR_SEND);
                // receive vectors
                subIndex.receiveData(recv_header.size);
                num_recv += recv_header.size;
            }
        }

        logger.num_elements = num_recv;
        subIndex.build(
            200, // ef_construction, TODO: hard coded
            16, // M_sub, TODO: hard coded
            8 // num_building_threads, TODO: hard coded
        );

        std::string filename = "executor_" + std::to_string(node) + "_" + dataset_name + "_" + std::to_string(num_partitions) + ".json";
        logger.sub_index_path = subIndex.save(filename);
        logger.saveExecutorLog(node);
    }

    MPI_Finalize();
}