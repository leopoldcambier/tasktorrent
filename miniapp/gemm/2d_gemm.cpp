#include "../utils_shared.hpp"

/**
 * Matrix is of size N (global)
 * Each rank works on Nr x Nr
 * Each thread works on Nt x Nt
 * 
 * There are n sub blocks on a given rank
 */
void gemm(const int matrix_size, const int block_size, const int n_threads, int nprows, int npcols, const bool test, const bool use_large)
{
    const int rank = ttor::comms_world_rank();
    const int n_ranks = ttor::comms_world_size();
    assert(matrix_size % block_size == 0);
    const int num_blocks = matrix_size / block_size;
    const int verb = 0;

    // You may want to do that to avoid the slow first call to MKL (??)
    warmup_mkl(n_threads);

    std::vector<Eigen::MatrixXd> A_ij(num_blocks * num_blocks, Eigen::MatrixXd());
    std::vector<Eigen::MatrixXd> C_ij(num_blocks * num_blocks, Eigen::MatrixXd());
    std::vector<Eigen::MatrixXd> B_ij(num_blocks * num_blocks, Eigen::MatrixXd());

    auto block_2_rank = [&](int i, int j) { return (i % nprows) + (j % npcols) * nprows; };
    const int rank_i = rank % nprows;
    const int rank_j = rank / nprows; 
    const int first_row_rank = rank_i;
    const int first_col_rank = rank_j * nprows;

    for(int i = 0; i < num_blocks; i++) {
        for(int j = 0; j < num_blocks; j++) {
            auto val = [&](int i_, int j_) { return val_global(i * block_size + i_, j * block_size + j_); };
            if(block_2_rank(i,j) == rank) {
                A_ij[i + j*num_blocks] = Eigen::MatrixXd::NullaryExpr(block_size, block_size, val);
                B_ij[i + j*num_blocks] = Eigen::MatrixXd::NullaryExpr(block_size, block_size, val);
                C_ij[i + j*num_blocks] = Eigen::MatrixXd::Zero(block_size, block_size);
            }
        }
    }
    
    /**
     * Initialize the runtime structures
     **/
    auto comm = ttor::make_communicator_world(verb);
    ttor::Threadpool tp(n_threads, comm.get(), verb, "Wk_Gemm_" + std::to_string(rank) + "_");

    ttor::Taskflow<int2> send_Aij(&tp, verb); // Send A_ij
    ttor::Taskflow<int2> send_Bij(&tp, verb); // Send B_ij
    ttor::Taskflow<int3> gemm_Cikj(&tp, verb); // += A_ik * B_kj
            
    /** 
     * Send
     **/

    auto send_Aij_am_large = comm->make_large_active_msg([&](int& i, int& j) {
        assert(block_2_rank(i,j) != rank);
        for(int k = rank_j; k < num_blocks; k += npcols) {
            assert(block_2_rank(i,k) == rank);
            gemm_Cikj.fulfill_promise({i, j, k});
        }
    }, [&](int& i, int& j) {
        A_ij[i + j*num_blocks].resize(block_size, block_size);
        return A_ij[i + j*num_blocks].data();
    }, [&](int& i, int& j){});

    auto send_Aij_am = comm->make_active_msg([&](ttor::view<double>& Aij, int& i, int& j) {
        assert(block_2_rank(i,j) != rank);
        A_ij[i + j*num_blocks].resize(block_size, block_size);
        copy_from_view(&A_ij[i + j*num_blocks], Aij);
        for(int k = rank_j; k < num_blocks; k += npcols) {
            assert(block_2_rank(i,k) == rank);
            gemm_Cikj.fulfill_promise({i, j, k});
        }
    });

    auto send_Bij_am_large = comm->make_large_active_msg([&](int& i, int& j) {
        assert(block_2_rank(i,j) != rank);
        for(int k = rank_i; k < num_blocks; k += nprows) {
            assert(block_2_rank(k,j) == rank);
            gemm_Cikj.fulfill_promise({k, i, j});
        }
    }, [&](int& i, int& j) {
        B_ij[i + j*num_blocks].resize(block_size, block_size);
        return B_ij[i + j*num_blocks].data();
    }, [&](int& i, int& j){});

    auto send_Bij_am = comm->make_active_msg([&](ttor::view<double>& Bij, int& i, int& j) {
        assert(block_2_rank(i,j) != rank);
        B_ij[i + j*num_blocks].resize(block_size, block_size);
        copy_from_view(&B_ij[i + j*num_blocks], Bij);
        for(int k = rank_i; k < num_blocks; k += nprows) {
            assert(block_2_rank(k,j) == rank);
            gemm_Cikj.fulfill_promise({k, i, j});
        }
    });

    send_Aij.set_task([&](int2 ij){
        int i = ij[0];
        int j = ij[1];
        assert(block_2_rank(i,j) == rank);
        ttor::view<double> A_view = make_view(&A_ij[i + j*num_blocks]);
        // Send A
	for(int r = first_row_rank; r < n_ranks; r += nprows) {
            if(r == rank) {
                for(int k = rank_j; k < num_blocks; k += npcols) gemm_Cikj.fulfill_promise({i,j,k});
            } else {
                if(use_large) send_Aij_am_large->send_large(r, A_view, i, j);
                else          send_Aij_am->send(r, A_view, i, j);
            }
	}
    }).set_indegree([&](int2) {
        return 1;
    }).set_priority([&](int2 ij) {
        return num_blocks - ij[1];
    }).set_mapping([&](int2 ij) {
        return (ij[0] / nprows + ij[1] / npcols * (num_blocks / nprows)) % n_threads;
    }).set_name([&](int2 ij) {
        return "send_A_" + std::to_string(ij[0]) + "_" + std::to_string(ij[1]);
    });

    send_Bij.set_task([&](int2 ij){
        int i = ij[0];
        int j = ij[1];
        assert(block_2_rank(i,j) == rank);
        ttor::view<double> B_view = make_view(&B_ij[i + j*num_blocks]);
        // Send B
        for(int r = first_col_rank; r < first_col_rank + nprows; r += 1) {
            if(r == rank) {
                for(int k = rank_i; k < num_blocks; k += nprows) gemm_Cikj.fulfill_promise({k,i,j});
            } else {
                if(use_large) send_Bij_am_large->send_large(r, B_view, i, j);
                else          send_Bij_am->send(r, B_view, i, j);
            }
        }
    }).set_indegree([&](int2) {
        return 1;
    }).set_priority([&](int2 ij) {
        return num_blocks - ij[0];
    }).set_mapping([&](int2 ij) {
        return (ij[0] / nprows + ij[1] / npcols * (num_blocks / nprows)) % n_threads;
    }).set_name([&](int2 ij) {
        return "send_B_" + std::to_string(ij[0]) + "_" + std::to_string(ij[1]);
    });

    /** 
     * GEMM
     **/

    // (ikj) compute C_ij += A_ik * B_kj
    gemm_Cikj.set_task([&](int3 ikj){
        int i = ikj[0];
        int k = ikj[1];
        int j = ikj[2];
        assert(block_2_rank(i,j) == rank);
        ttor::timer t0 = ttor::wctime();
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, block_size, block_size, block_size, 1.0, A_ij[i + k * num_blocks].data(), block_size, B_ij[k + j * num_blocks].data(), block_size, 1.0, C_ij[i + j * num_blocks].data(), block_size);
        ttor::timer t1 = ttor::wctime();
        if(k < num_blocks-1) {
            gemm_Cikj.fulfill_promise({i,k+1,j});
        }
    }).set_indegree([&](int3 ikj) {
        return (ikj[1] == 0 ? 2 : 3);
    }).set_mapping([&](int3 ikj) {
        return (ikj[0] / nprows + ikj[2] / npcols * (num_blocks / nprows)) % n_threads;
    }).set_name([&](int3 ikj) {
        return "prod_C_" + std::to_string(ikj[0]) + "_" + std::to_string(ikj[1]) + "_" + std::to_string(ikj[2]);
    });


    ttor::comms_world_barrier();
    printf("Starting 2D Gemm...\n");
    ttor::timer t0 = ttor::wctime();
    for(int i = 0; i < num_blocks; i++) {
        for(int j = 0; j < num_blocks; j++) {
            if(block_2_rank(i,j) == rank) {
                send_Aij.fulfill_promise({i,j});
                send_Bij.fulfill_promise({i,j});
            }
        }
    }
    ttor::timer t_overhead = ttor::wctime();
    tp.join();
    ttor::comms_world_barrier();
    ttor::timer t1 = ttor::wctime();
    double total_time = ttor::elapsed(t0, t1);
    long long int flops_per_rank = ((long long int)matrix_size) * ((long long int)matrix_size) * ((long long int)matrix_size) / ((long long int)n_ranks);
    long long int flops_per_core = flops_per_rank / ((long long int)n_threads);
    // For easy CSV parsing
    printf("%e\n", ttor::elapsed(t0, t_overhead));
    printf("[rank]>>>>rank n_ranks n_cores nthreads nprows npcols use_large matrix_size num_blocks block_size total_time flops_per_core flops_per_rank\n");
    printf("[%d]>>>>ttor_2d_gemm %d %d %d %d %d %d %d %d %d %d %e %llu %llu\n",rank,rank,n_ranks,n_ranks*n_threads,n_threads,nprows,npcols,use_large,matrix_size,num_blocks,block_size,total_time,flops_per_core,flops_per_rank);
                
    if(test) {
        printf("Testing..\n");
        // Send all to 0
        int n_received = 0;
        int n_expected = (rank == 0 ? num_blocks * num_blocks : 0);
        Eigen::MatrixXd C_test = Eigen::MatrixXd::Zero(matrix_size, matrix_size);
        auto comm = ttor::make_communicator_world(verb);
        auto am = comm->make_active_msg([&](ttor::view<double>& A, int& i, int& j) {
            C_test.block(i * block_size, j * block_size, block_size, block_size) = make_from_view(A, block_size);
            n_received++;
        });
        for(int i = 0; i < num_blocks; i++) {
            for(int j = 0; j < num_blocks; j++) {
                if(block_2_rank(i, j) == rank) {
                    auto C_view = make_view(&C_ij[i + j * num_blocks]);
                    am->send(0, C_view, i, j);
                }
            }
        }
        while((!comm->is_done()) || (n_received < n_expected)) {
            comm->progress();
        }
        // Compute reference on 0
        if(rank == 0) {
            Eigen::MatrixXd A_ref = Eigen::MatrixXd::NullaryExpr(matrix_size, matrix_size, [](int i, int j){return val_global(i,j);});
            Eigen::MatrixXd B_ref = Eigen::MatrixXd::NullaryExpr(matrix_size, matrix_size, [](int i, int j){return val_global(i,j);});
            ttor::timer t0 = ttor::wctime();
            Eigen::MatrixXd C_ref = A_ref * B_ref;
            ttor::timer t1 = ttor::wctime();
            double error = (C_ref - C_test).norm() / C_ref.norm();
            printf("\n==> GEMM error %e\n\n", error);
            printf("Reference code took %e\n", ttor::elapsed(t0, t1));
            if(error > 1e-12) {
                exit(1);
            }
        }
    }
}

int main(int argc, const char **argv)
{
    ttor::comms_init();

    std::stringstream sstr;
    sstr << ttor::comms_world_size();
    const std::string comm_size_str = sstr.str();

    cxxopts::Options options("2d_gemm", "2D dense gemm using TaskTorrent");
    options.add_options()
        ("help", "Print help")
        ("n_threads", "Number of threads", cxxopts::value<int>()->default_value("1"))
        ("verb", "Verbosity level", cxxopts::value<int>()->default_value("0"))
        ("block_size", "Block size", cxxopts::value<int>()->default_value("8"))
        ("matrix_size", "Total matrix size", cxxopts::value<int>()->default_value("128"))
        ("nprows", "Number of processors accross rows", cxxopts::value<int>()->default_value("1"))
        ("npcols", "Number of processors accross columns", cxxopts::value<int>()->default_value(comm_size_str.c_str()))
        ("test", "Test or not", cxxopts::value<bool>()->default_value("true"))
        ("use_large", "Use large active messages", cxxopts::value<bool>()->default_value("true"));
    auto result = options.parse(argc, argv);

    const int n_threads = result["n_threads"].as<int>();
    const int verb = result["verb"].as<int>();
    const int block_size = result["block_size"].as<int>();
    const int matrix_size = result["matrix_size"].as<int>();
    const int nprows = result["nprows"].as<int>();
    const int npcols = result["npcols"].as<int>();
    const bool test = result["test"].as<bool>();
    const bool use_large = result["use_large"].as<bool>();

    assert(block_size > 0);
    assert(matrix_size > 0);
    assert(n_threads > 0);
    assert(verb >= 0);
    assert(nprows >= 0);
    assert(npcols >= 0);

    if (result.count("help")) {
        std::cout << options.help({"", "Group"}) << std::endl;
        exit(0);
    }
    if(ttor::comms_world_rank() == 0) printf("Arguments: block_size %d\nmatrix_size %d\nn_threads %d\nverb %d\nnprows %d\nnpcols %d\ntest %d\nuse_large %d\n", 
        block_size, matrix_size, n_threads, verb, nprows, npcols, test, use_large);

    gemm(matrix_size, block_size, n_threads, nprows, npcols, test, use_large);

    ttor::comms_finalize();
}
