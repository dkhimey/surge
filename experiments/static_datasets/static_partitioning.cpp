// static_partitioning.cpp
//
// Builds the static distributed index: a k-means routing layer partitioned with
// KaHIP, then distributes the vectors and builds each worker's local HNSW shard.
// Saves the coordinator and per-worker indices to <dataset>_<num_partitions>/,
// which static_qps then loads.
//
// Usage:  mpirun -np <P+1> ./static_partitioning <dataset> <num_partitions>

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
    std::string log_dir = ensure_log_dir(log_id);

    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    install_mpi_terminate_handler();

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

    int nvectors, dim;
    printf("Max threads: %d\n", omp_get_max_threads());

    if (node == 0) { // Coordinator
        // take sample (just read in first sample_size elements, assume they are randomly distributed across file)
        std::pair<int,int> data_info = get_dataset_info(DATASETS[dataset_name]["base_file"]);
        int nvectors = data_info.first;
        int dim = data_info.second;

        comm.broadcast_dataset_info(nvectors, dim, world_size);

        size_t sample_size = 100000; //TODO: hard coded

        std::vector<float> sample = getSample(DATASETS[dataset_name]["base_file"], 
                                              nvectors, dim, sample_size);
        
        // build meta hnsw on the centers
        Coordinator metaIndex(dim, &comm);
        metaIndex.set_sample_data(sample.data(), sample_size);

        int ncenters = 10000; //TODO: hard coded
        int ef_construction = 200; //TODO: hard coded
        int M_meta = 16; //TODO: hard coded

        metaIndex.build(ncenters, num_partitions, ef_construction, M_meta);

        // assign vectors to workers
        double start = MPI_Wtime();
        int num_threads = omp_get_max_threads(); //TODO: hard coded
        std::vector<int> counts_per_partition = metaIndex.distribute_vectors(
            DATASETS[dataset_name]["base_file"],
            nvectors,
            num_threads
        );
        double end = MPI_Wtime();

        double distribute_time = end - start;
        std::cout << "Partition time: " << distribute_time << " seconds\n";

        // complete build process from Coordinator side
        comm.broadcast_termination(world_size);

        std::string meta_dir = dataset_name + "_" + std::to_string(num_partitions);
        metaIndex.save(meta_dir);

        write_controller_build_json(log_dir + "/controller_build.json", metaIndex.build_metrics(),
                                    distribute_time, counts_per_partition, meta_dir + "/metaHNSW.bin");
        dump_centers(log_dir, metaIndex.centers());
        dump_partitions(log_dir, metaIndex.get_partitions());
    } else {
        std::cout << "[Executor " << node << "] log_id: " << log_id << "\n";
        comm.recv_dataset_info(nvectors, dim);
        std::cout << "[Executor " << node << "] Received dataset info: num vectors = " << nvectors << ", dimension = " << dim << "\n";

        Executor subIndex(node, dim, comm);

        size_t num_recv = 0;
        bool done = false;

        while (!done) {
            // recieve header
            MessageHeader recv_header;
            comm.recv_header(recv_header, 0);
            // std::cout << "[Executor " << node << "] Received header: type=" << recv_header.type << ", size=" << recv_header.size << "\n";
            if (recv_header.type == END_OF_COMMUNICATION) {
                done = true;
            } else {
                assert(recv_header.type == VECTOR_SEND);
                // receive vectors
                subIndex.receive_data(recv_header.size);
                num_recv += recv_header.size;
            }
        }

        std::cout << "[Executor " << node << "] Total vectors received: " << num_recv << "\n";
        subIndex.build(
            200, // ef_construction, TODO: hard coded
            16, // M_sub, TODO: hard coded
            omp_get_max_threads() // num_building_threads, TODO: hard coded
        );

        std::string output_dir = dataset_name + "_" + std::to_string(num_partitions);
        std::filesystem::create_directories(output_dir);

        std::string filename_prefix = output_dir + "/executor_" + std::to_string(node) + "_" + dataset_name + "_" + std::to_string(num_partitions);
        std::string sub_file = subIndex.save(filename_prefix);
        write_executor_build_json(log_dir + "/executor_" + std::to_string(node) + "_build.json",
                                  subIndex.build_metrics(), num_recv, sub_file);
    }

    MPI_Finalize();
}

// mpirun -np 11 --rankfile rankfile.txt bin/static_partitioning bigann-500M 10