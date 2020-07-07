#include "gemm_shared.hpp"

/**
 * Matrix is of size N (global)
 * Each rank works on Nr x Nr
 * Each thread works on Nt x Nt
 * 
 * There are n sub blocks on a given rank
 */
void gemm(const int matrix_size, const int block_size, const int n_threads, int nprows, int npcols, const bool test, const bool use_large)
{
    const int rank = ttor::comm_rank();
    const int n_ranks = ttor::comm_size();
    assert(matrix_size % block_size == 0);
    const int num_blocks = matrix_size / block_size;
    const int verb = 0;
    
    std::vector<Eigen::MatrixXd> A_ij(num_blocks * num_blocks, Eigen::MatrixXd());
    std::vector<Eigen::MatrixXd> C_ij(num_blocks * num_blocks, Eigen::MatrixXd());
    std::vector<Eigen::MatrixXd> B_ij(num_blocks * num_blocks, Eigen::MatrixXd());

    auto block_2_rank = [&](int i, int j) { return (i % nprows) + (j % npcols) * nprows; };

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
    ttor::Communicator comm(MPI_COMM_WORLD, verb);
    ttor::Threadpool tp(n_threads, &comm, verb, "Wk_Gemm_" + to_string(rank) + "_");

    ttor::Taskflow<int2> send_Aij(&tp, verb); // Send A_ij
    ttor::Taskflow<int2> send_Bij(&tp, verb); // Send B_ij
    ttor::Taskflow<int3> gemm_Cikj(&tp, verb); // += A_ik * B_kj

    /** 
     * Send
     **/

    auto send_Aij_am_large = comm.make_large_active_msg([&](int& i, int& j) {
        for(int k = 0; k < num_blocks; k++) {
            if(block_2_rank(i,k) == rank) {
                gemm_Cikj.fulfill_promise({i, j, k});
            }
        }
    }, [&](int& i, int& j) {
        A_ij[i + j*num_blocks].resize(block_size, block_size);
        return A_ij[i + j*num_blocks].data();
    }, [&](int& i, int& j){});

    auto send_Aij_am = comm.make_active_msg([&](ttor::view<double>& Aij, int& i, int& j) {
        A_ij[i + j*num_blocks].resize(block_size, block_size);
        copy_from_view(&A_ij[i + j*num_blocks], Aij);
        for(int k = 0; k < num_blocks; k++) {
            if(block_2_rank(i,k) == rank) {
                gemm_Cikj.fulfill_promise({i, j, k});
            }
        }
    });

    auto send_Bij_am_large = comm.make_large_active_msg([&](int& i, int& j) {
        for(int k = 0; k < num_blocks; k++) {
            if(block_2_rank(k,j) == rank) {
                gemm_Cikj.fulfill_promise({k, i, j});
            }
        }
    }, [&](int& i, int& j) {
        B_ij[i + j*num_blocks].resize(block_size, block_size);
        return B_ij[i + j*num_blocks].data();
    }, [&](int& i, int& j){});

    auto send_Bij_am = comm.make_active_msg([&](ttor::view<double>& Bij, int& i, int& j) {
        B_ij[i + j*num_blocks].resize(block_size, block_size);
        copy_from_view(&B_ij[i + j*num_blocks], Bij);
        for(int k = 0; k < num_blocks; k++) {
            if(block_2_rank(k,j) == rank) {
                gemm_Cikj.fulfill_promise({k, i, j});
            }
        }
    });

    send_Aij.set_task([&](int2 ij){
        int i = ij[0];
        int j = ij[1];
        assert(block_2_rank(i,j) == rank);
        ttor::view<double> A_view = make_view(&A_ij[i + j*num_blocks]);
        ttor::view<double> B_view = make_view(&B_ij[i + j*num_blocks]);
        // Send A
        std::set<int> dest_A;
        for(int k = 0; k < num_blocks; k++) { dest_A.insert(block_2_rank(i,k)); }
        for(int dest: dest_A) {
            if(dest == rank) {
                for(int k = 0; k < num_blocks; k++) {
                    if(block_2_rank(i,k) == rank) gemm_Cikj.fulfill_promise({i, j, k});
                }
            } else {
                if(use_large) send_Aij_am_large->send_large(dest, A_view, i, j);
                else          send_Aij_am->send(dest, A_view, i, j);
            }
        }        
    }).set_indegree([&](int2) {
        return 1;
    }).set_priority([&](int2 ij) {
        return num_blocks - ij[1];
    }).set_mapping([&](int2 ij) {
        return (ij[0] % n_threads);
    });

    send_Bij.set_task([&](int2 ij){
        int i = ij[0];
        int j = ij[1];
        assert(block_2_rank(i,j) == rank);
        ttor::view<double> B_view = make_view(&B_ij[i + j*num_blocks]);
        // Send B
        std::set<int> dest_B;
        for(int k = 0; k < num_blocks; k++) { dest_B.insert(block_2_rank(k,j)); }
        for(int dest: dest_B) {
            if(dest == rank) {
                for(int k = 0; k < num_blocks; k++) {
                    if(block_2_rank(k,j) == rank) gemm_Cikj.fulfill_promise({k, i, j});
                }
            } else {
                if(use_large) send_Bij_am_large->send_large(dest, B_view, i, j);
                else          send_Bij_am->send(dest, B_view, i, j);
            }
        }
    }).set_indegree([&](int2) {
        return 1;
    }).set_priority([&](int2 ij) {
        return num_blocks - ij[0];
    }).set_mapping([&](int2 ij) {
        return (ij[0] % n_threads);
    });

    /** 
     * GEMM
     **/

    // (ikj) compute C_ij += A_ik * B_kj
    gemm_Cikj.set_task([&](int3 ikj){
        int i = ikj[0];
        int k = ikj[1];
        int j = ikj[2];
        ttor::timer t0 = ttor::wctime();
        C_ij[i + j * num_blocks].noalias() += A_ij[i + k * num_blocks] * B_ij[k + j * num_blocks];
        ttor::timer t1 = ttor::wctime();
        if(k < num_blocks-1) {
            gemm_Cikj.fulfill_promise({i,k+1,j});
        }
    }).set_indegree([&](int3 ikj) {
        return (ikj[1] == 0 ? 2 : 3);
    }).set_mapping([&](int3 ikj) {
        return (ikj[0] / nprows + ikj[2] / npcols * (num_blocks / nprows)) % n_threads;
    });

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
    ttor::timer t1 = ttor::wctime();
    double total_time = ttor::elapsed(t0, t1);
    long long int flops_per_rank = ((long long int)matrix_size) * ((long long int)matrix_size) * ((long long int)matrix_size) / ((long long int)n_ranks);
    long long int flops_per_core = flops_per_rank / ((long long int)n_threads);
    // For easy CSV parsing
    printf("%e\n", ttor::elapsed(t0, t_overhead));
    printf("[rank]>>>>rank,n_ranks,nthreads,nprows,npcols,use_large,matrix_size,num_blocks,block_size,tot_time,flops_per_core,flops_per_rank\n");
    printf("[%d]>>>>ttor_2d_gemm %d %d %d %d %d %d %d %d %d %e %llu %llu\n",rank,rank,n_ranks,n_threads,nprows,npcols,use_large,matrix_size,num_blocks,block_size,total_time,flops_per_core,flops_per_rank);

    MPI_Barrier(MPI_COMM_WORLD);
    if(test) {
        printf("Testing..\n");
        // Send all to 0
        int n_received = 0;
        int n_expected = (rank == 0 ? num_blocks * num_blocks : 0);
        Eigen::MatrixXd C_test = Eigen::MatrixXd::Zero(matrix_size, matrix_size);
        ttor::Communicator comm(MPI_COMM_WORLD, verb);
        auto am = comm.make_active_msg([&](ttor::view<double>& A, int& i, int& j) {
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
        while((!comm.is_done()) || (n_received < n_expected)) {
            comm.progress();
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
        }
    }
}

int main(int argc, char **argv)
{
    int req = MPI_THREAD_FUNNELED;
    int prov = -1;

    MPI_Init_thread(NULL, NULL, req, &prov);

    assert(prov == req);

    int N = 128;
    int Nt = 8;
    int n_threads = 1;
    int verb = 0;
    bool test = true;
    std::string logfile = "";
    int nprows = 1;
    int npcols = 1;
    bool use_large = true;

    if (argc >= 2)
    {
        N = atoi(argv[1]);
        assert(N > 0);
    }
    
    if (argc >= 3) {
        Nt = atoi(argv[2]);
        assert(Nt > 0);
        assert(Nt <= N);
    }

    if (argc >= 4) {
        n_threads = atoi(argv[3]);
        assert(n_threads > 0);
    }

    if (argc >= 6) {
        nprows = atoi(argv[4]);
        npcols = atoi(argv[5]);
    }

    if (argc >= 7) {
        test = static_cast<bool>(atoi(argv[6]));
    }

    if (argc >= 8) {
        use_large = static_cast<bool>(atoi(argv[7]));
    }

    if(ttor::comm_rank() == 0) printf("Usage: ./2d_gemm matrix_size block_size n_threads nprows npcols test use_large\n");

    gemm(N, Nt, n_threads, nprows, npcols, test, use_large);

    MPI_Finalize();
}
