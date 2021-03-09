#include "../utils_shared.hpp"

/**
 * Matrix is of size N (global)
 * Each rank works on Nr x Nr
 * Each thread works on Nt x Nt
 * 
 * There are n sub blocks on a given rank
 */
void gemm(const int N, const int Nt, const int n_threads, std::string logfile, const int verb, const bool test)
{
    const int rank = ttor::comms_world_rank();
    const int n_ranks = ttor::comms_world_size();
    const int n_ranks_1d = static_cast<int>(round(pow(n_ranks, 1.0/3.0)));
    assert(n_ranks_1d * n_ranks_1d * n_ranks_1d == n_ranks);
    const int rank_i = rank % n_ranks_1d;
    const int rank_j = (rank / n_ranks_1d) % n_ranks_1d;
    const int rank_k = rank / (n_ranks_1d * n_ranks_1d);
    const int3 rank_ijk = {rank_i, rank_j, rank_k};
    const int Nr = N / n_ranks_1d;
    assert(Nr * n_ranks_1d == N);
    const int n = Nr / Nt;
    assert(Nt * n == Nr);
    printf("Hello rank %d with 3d-index (%d %d %d) / (%d %d %d), N %d, Nr %d, Nt %d, n %d\n", rank, rank_i, rank_j, rank_k, n_ranks_1d, n_ranks_1d, n_ranks_1d, 
        N, Nr, Nt, n);

    printf("rank,%d\n", rank);
    printf("rank_i,%d\n", rank_i);
    printf("rank_j,%d\n", rank_j);
    printf("rank_k,%d\n", rank_k);
    printf("ntot,%d\n", N);
    printf("nrank,%d\n", Nr);
    printf("ntile,%d\n", Nt);
    printf("nthreads,%d\n", n_threads);
    printf("logfile,%s\n", logfile.c_str());
    printf("verb,%d\n", verb);
    printf("test,%d\n", test);
    
    // You may want to do that to avoid the slow first call to MKL (??)
    warmup_mkl(n_threads);
    
    auto rank_ijk_to_rank = [n_ranks_1d](int rank_i, int rank_j, int rank_k) {
        return rank_k * n_ranks_1d * n_ranks_1d + rank_j * n_ranks_1d + rank_i;
    };
    assert(rank_ijk_to_rank(rank_i, rank_j, rank_k) == rank);

    /**
     * Record timings
     **/
    std::atomic<long long int> send_copy_us_t(0);
    std::atomic<long long int> send_am_us_t(0);
    std::atomic<long long int> bcst_copy_us_t(0);
    std::atomic<long long int> bcst_am_us_t(0);
    std::atomic<long long int> gemm_us_t(0);
    std::atomic<long long int> gemm_copy_us_t(0);
    std::atomic<long long int> gemm_am_us_t(0);
    std::atomic<long long int> accu_us_t(0);

    /**
     * Original and final matrices
     **/
    // n x n matrix of Nt x Nt matrices, so Nr x Nr total
    std::vector<std::vector<Eigen::MatrixXd>> A_ij(n, std::vector<Eigen::MatrixXd>(n, Eigen::MatrixXd::Zero(Nt, Nt)));
    std::vector<std::vector<Eigen::MatrixXd>> C_ij(n, std::vector<Eigen::MatrixXd>(n, Eigen::MatrixXd::Zero(Nt, Nt)));
    std::vector<std::vector<Eigen::MatrixXd>> B_ij(n, std::vector<Eigen::MatrixXd>(n, Eigen::MatrixXd::Zero(Nt, Nt)));

    if(rank_k == 0) {
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                auto val = [&](int i_, int j_) { return val_global(rank_i * Nr + i * Nt + i_, rank_j * Nr + j * Nt + j_); };
                A_ij[i][j] = Eigen::MatrixXd::NullaryExpr(Nt, Nt, val);
                B_ij[i][j] = Eigen::MatrixXd::NullaryExpr(Nt, Nt, val);
            }
        }
    }

    /** 
     * Workspace
     **/
    std::vector<std::vector<Eigen::MatrixXd>> A_ijk(n, std::vector<Eigen::MatrixXd>(n, Eigen::MatrixXd::Zero(0, 0)));
    std::vector<std::vector<Eigen::MatrixXd>> C_ijk(n, std::vector<Eigen::MatrixXd>(n, Eigen::MatrixXd::Zero(0, 0)));
    std::vector<std::vector<Eigen::MatrixXd>> B_ijk(n, std::vector<Eigen::MatrixXd>(n, Eigen::MatrixXd::Zero(0, 0)));
    
    // C_ijk_accu[sub_i][sub_j][from] stores the results for (sub_i, sub_j) to be accumulated, from rank from
    std::vector<std::vector<std::vector<Eigen::MatrixXd>>> C_ijk_accu(
        n, std::vector<std::vector<Eigen::MatrixXd>>(
        n, std::vector<Eigen::MatrixXd>(
        n_ranks_1d, Eigen::MatrixXd(0, 0))));
    
    /**
     * Initialize the runtime structures
     **/
    auto comm = ttor::make_communicator_world(verb);
    ttor::Threadpool tp(n_threads, comm.get(), verb, "Wk_Gemm_" + std::to_string(rank) + "_");

    // send is indexed by int2, which are the sub blocks
    ttor::Taskflow<int2> send_Aij(&tp, verb);  // (i,j,0) sends A_ij to (i,j,j) for all i,j
    ttor::Taskflow<int2> send_Bij(&tp, verb);  // (i,j,0) sends B_ij to (i,j,i) for all i,j
    // send is indexed by int2, which are the sub blocks
    ttor::Taskflow<int2> bcst_Aij(&tp, verb);  // (i,j,j) sends A_ij along j to all (i,*,j) for all i,j
    ttor::Taskflow<int2> bcst_Bij(&tp, verb);  // (i,j,i) sends B_ij along i to all (*,j,i) for all i,j
    // gemm is indexed by int3, which are the sub blocks
    ttor::Taskflow<int3> gemm_Cijk(&tp, verb); // (i,j,k) compute C_ijk = A_ik * B_kj, send for accumulation reduction on (i,j,0)
    ttor::Taskflow<int3> accu_Cij(&tp, verb);  // accumulate (i,j,from) into (i,j)

    ttor::Logger log(1000000);
    if(logfile.size() > 0) {
        tp.set_logger(&log);
        comm->set_logger(&log);
    }

    /** 
     * Send
     **/

    auto send_Aij_am = comm->make_large_active_msg([&](const int& sub_i, const int& sub_j) {
        bcst_Aij.fulfill_promise({sub_i, sub_j});
    }, [&](const int& sub_i, const int& sub_j) {
        A_ijk[sub_i][sub_j].resize(Nt, Nt);
        return A_ijk[sub_i][sub_j].data();
    }, [&](const int& sub_i, const int& sub_j){});

    auto send_Bij_am = comm->make_large_active_msg([&](const int& sub_i, const int& sub_j) {
        bcst_Bij.fulfill_promise({sub_i, sub_j});
    }, [&](const int& sub_i, const int& sub_j) {
        B_ijk[sub_i][sub_j].resize(Nt, Nt);
        return B_ijk[sub_i][sub_j].data();
    }, [&](const int& sub_i, const int& sub_j){});

    // (i,j,0) sends A_ij to (i,j,j) for all i,j
    send_Aij.set_task([&](int2 sub_ij){
        assert(rank_k == 0);
        scoped_timer t(&send_copy_us_t);
        int sub_i = sub_ij[0];
        int sub_j = sub_ij[1];
        ttor::view<double> A_view = make_view(&A_ij[sub_i][sub_j]);
        int dest = rank_ijk_to_rank(rank_i, rank_j, rank_j);
        if(dest == rank) {
            A_ijk[sub_i][sub_j] = A_ij[sub_i][sub_j];
            bcst_Aij.fulfill_promise({sub_i, sub_j});
        } else {
            send_Aij_am->send_large(dest, A_view, sub_i, sub_j);
        }
    }).set_indegree([&](int2) {
        return 1;
    }).set_mapping([&](int2 sub_ij) {
        return sub_ij[0] % n_threads;
    }).set_priority([&](int2 sub_ij){
        return 1.0 * n + (n - sub_ij[1]);
    }).set_binding([&](int2 sub_ij){
        return false;
    }).set_name([&](int2 sub_ij) { return "send_A_" + to_string(sub_ij) + "_" + to_string(rank_ijk); });

    // (i,j,0) sends B_ij to (i,j,i) for all i,j
    send_Bij.set_task([&](int2 sub_ij){
        assert(rank_k == 0);
        scoped_timer t(&send_copy_us_t);
        int sub_i = sub_ij[0];
        int sub_j = sub_ij[1];
        ttor::view<double> B_view = make_view(&B_ij[sub_i][sub_j]);
        int dest = rank_ijk_to_rank(rank_i, rank_j, rank_i);
        if(dest == rank) {
            B_ijk[sub_i][sub_j] = B_ij[sub_i][sub_j];
            bcst_Bij.fulfill_promise({sub_i, sub_j});
        } else {
            send_Bij_am->send_large(dest, B_view, sub_i, sub_j);
        }
    }).set_indegree([&](int2) {
        return 1;
    }).set_mapping([&](int2 sub_ij) {
        return (sub_ij[1] % n_threads);
    }).set_priority([&](int2 sub_ij){
        return 1.0 * n + (n - sub_ij[0]);
    }).set_binding([&](int2 sub_ij){
        return false;
    }).set_name([&](int2 sub_ij) { return "send_B_" + to_string(sub_ij) + "_" + to_string(rank_ijk); });

    /** 
     * Broadcast
     **/

    auto bcst_Aij_am = comm->make_large_active_msg([&](const int &sub_i, const int &sub_j) {
        for(int k = 0; k < n; k++) {
            gemm_Cijk.fulfill_promise({sub_i, k, sub_j});
        }
    }, [&](const int &sub_i, const int &sub_j) {
        A_ijk[sub_i][sub_j].resize(Nt, Nt);
        return A_ijk[sub_i][sub_j].data();
    }, [&](const int &sub_i, const int &sub_j) {});

    auto bcst_Bij_am = comm->make_large_active_msg([&](const int &sub_i, const int &sub_j) {
        for(int k = 0; k < n; k++) {
            gemm_Cijk.fulfill_promise({k, sub_j, sub_i});
        }
    }, [&](const int &sub_i, const int &sub_j) {
        B_ijk[sub_i][sub_j].resize(Nt, Nt);
        return B_ijk[sub_i][sub_j].data();
    }, [&](const int &sub_i, const int &sub_j) {});

    // (i,j,j) sends A_ij along j to all (i,*,j) for all i,j
    bcst_Aij.set_task([&](int2 sub_ij){
        scoped_timer t(&bcst_copy_us_t);
        assert(rank_j == rank_k);
        int sub_i = sub_ij[0];
        int sub_j = sub_ij[1];
        ttor::view<double> A_view = make_view(&A_ijk[sub_i][sub_j]);
        for(int k = 0; k < n_ranks_1d; k++) {
            int dest = rank_ijk_to_rank(rank_i, k, rank_j);
            if(dest == rank) {
                for(int l = 0; l < n; l++) {
                    gemm_Cijk.fulfill_promise({sub_i, l, sub_j});
                }
            } else {
                bcst_Aij_am->send_large(dest, A_view, sub_i, sub_j);
            }
        }
    }).set_indegree([&](int2) {
        return 1;
    }).set_mapping([&](int2 sub_ij) {
        return (sub_ij[0] % n_threads);
    }).set_priority([&](int2 sub_ij){
        return 1.0 * n + (n - sub_ij[1]);
    }).set_binding([&](int2 sub_ij){
        return false;
    }).set_name([&](int2 sub_ij) { return "bcast_A_" + to_string(sub_ij) + "_" + to_string(rank_ijk); });

    // (i,j,i) sends B_ij along i to all (*,j,i) for all i,j
    bcst_Bij.set_task([&](int2 sub_ij){
        scoped_timer t(&bcst_copy_us_t);
        assert(rank_i == rank_k);
        int sub_i = sub_ij[0];
        int sub_j = sub_ij[1];
        ttor::view<double> B_view = make_view(&B_ijk[sub_i][sub_j]);
        for(int k = 0; k < n_ranks_1d; k++) {
            int dest = rank_ijk_to_rank(k, rank_j, rank_i);
            if(dest == rank) {
                for(int l = 0; l < n; l++)
                    gemm_Cijk.fulfill_promise({l, sub_j, sub_i});
            } else {
                bcst_Bij_am->send_large(dest, B_view, sub_i, sub_j);
            }
        }
    }).set_indegree([&](int2) {
        return 1;
    }).set_mapping([&](int2 sub_ij) {
        return (sub_ij[1] % n_threads);
    }).set_priority([&](int2 sub_ij){
        return 1.0 * n + (n - sub_ij[0]);
    }).set_binding([&](int2 sub_ij){
        return false;
    }).set_name([&](int2 sub_ij) { return "bcast_B_" + to_string(sub_ij) + "_" + to_string(rank_ijk); });

    /** 
     * GEMM
     **/

    auto gemm_Cijk_am = comm->make_large_active_msg([&](const int &sub_i, const int &sub_j, const int& from) {
        accu_Cij.fulfill_promise({sub_i, sub_j, from});
    }, [&](const int& sub_i, const int& sub_j, const int& from) {
        C_ijk_accu[sub_i][sub_j][from].resize(Nt, Nt);
        return C_ijk_accu[sub_i][sub_j][from].data();
    }, [&](const int& sub_i, const int& sub_j, const int& from) {});

    // (i,j,k) compute C_ijk = A_ik * B_kj
    gemm_Cijk.set_task([&](int3 sub_ijk){
        int sub_i = sub_ijk[0];
        int sub_j = sub_ijk[1];
        int sub_k = sub_ijk[2];
        {
            scoped_timer t(&gemm_us_t);
            if(C_ijk[sub_i][sub_j].size() == 0) C_ijk[sub_i][sub_j] = Eigen::MatrixXd::Zero(Nt, Nt);
            C_ijk[sub_i][sub_j].noalias() += A_ijk[sub_i][sub_k] * B_ijk[sub_k][sub_j];
        }
        scoped_timer t(&gemm_copy_us_t);
        if(sub_ijk[2] < n-1) {
            gemm_Cijk.fulfill_promise({sub_i, sub_j, sub_k+1});
        } else {
            auto C_ijk_view = make_view(&C_ijk[sub_i][sub_j]);
            int dest = rank_ijk_to_rank(rank_i, rank_j, 0);
            if(dest == rank) {
                C_ijk_accu[sub_i][sub_j][rank_k] = C_ijk[sub_i][sub_j];
                accu_Cij.fulfill_promise({sub_i, sub_j, rank_k});
            } else {
                int k = rank_k;
                gemm_Cijk_am->send_large(dest, C_ijk_view, sub_i, sub_j, k);
            }
        }
    }).set_indegree([&](int3 sub_ijk) {
        return sub_ijk[2] == 0 ? 2 : 3; // 2 A_ik and B_kj blocks, + previous gemm
    }).set_mapping([&](int3 sub_ijk) {
        return (sub_ijk[0] + sub_ijk[1] * n + sub_ijk[2] * n * n) % n_threads;
    }).set_binding([&](int3 sub_ijk) {
        return false;
    }).set_priority([&](int3 sub_ijk){
        return 0.0 * n + (n - sub_ijk[2]);
    }).set_name([&](int3 sub_ijk) { return "gemm_C_" + to_string(sub_ijk) + "_" + to_string(rank_ijk); });

    // (i,j,k) compute C_ijk = A_ik * B_kj
    accu_Cij.set_task([&](int3 sub_ij_from){
        scoped_timer t(&accu_us_t);
        int sub_i = sub_ij_from[0];
        int sub_j = sub_ij_from[1];
        int from  = sub_ij_from[2];
        C_ij[sub_i][sub_j].noalias() += C_ijk_accu[sub_i][sub_j][from];
    }).set_indegree([&](int3) {
        return 1;
    }).set_mapping([&](int3 sub_ij_from) {
        return (sub_ij_from[0] + n * sub_ij_from[1]) % n_threads;
    }).set_binding([&](int3) {
        return true;
    }).set_priority([&](int3){
        return 0.0;
    }).set_name([&](int3 sub_ij_from) { return "accu_C_" + to_string(sub_ij_from) + "_" + to_string(rank_ijk); });

    printf("Starting 3D Gemm...\n");
    ttor::comms_world_barrier();
    ttor::timer t0 = ttor::wctime();
    if(rank_k == 0) {
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                send_Aij.fulfill_promise({i,j});
                send_Bij.fulfill_promise({i,j});
            }
        }
    }
    tp.join();
    ttor::comms_world_barrier();
    ttor::timer t1 = ttor::wctime();
    double total_time = ttor::elapsed(t0, t1);
    double gemm_time = gemm_us_t.load() * 1e-6;
    double gemm_time_per_thread = gemm_time / n_threads;
    printf("Done\n");
    printf("total_time,%e\n", ttor::elapsed(t0, t1));
    printf("send_copy_us_t,%e\n",send_copy_us_t.load() * 1e-6);
    printf("send_am_us_t,%e\n",send_am_us_t.load() * 1e-6);
    printf("bcst_copy_us_t,%e\n",bcst_copy_us_t.load() * 1e-6);
    printf("bcst_am_us_t,%e\n",bcst_am_us_t.load() * 1e-6);
    printf("gemm_us_t,%e\n",gemm_us_t.load() * 1e-6);
    printf("gemm_copy_us_t,%e\n",gemm_copy_us_t.load() * 1e-6);
    printf("gemm_am_us_t,%e\n",gemm_am_us_t.load() * 1e-6);
    printf("accu_us_t,%e\n",accu_us_t.load() * 1e-6);
    // For easy CSV parsing
    long long int flops_per_rank = (long long int)(N) * (long long int)(N) * (long long int)(N) / ((long long int)n_ranks);
    long long int flops_per_core = flops_per_rank / ((long long int)n_threads);
    printf("[rank]>>>>matrix_size rank_block_size block_size rank n_ranks n_cores nthreads total_time gemm_time gemm_time_per_thread flops_per_core flops_per_rank\n");
    printf("[%d]>>>>ttor_3d_gemm %d %d %d %d %d %d %d %e %e %e %llu %llu\n",rank,N,Nr,Nt,rank,n_ranks,n_ranks*n_threads,n_threads,total_time,gemm_time,gemm_time_per_thread,flops_per_core,flops_per_rank);

    if(logfile.size() > 0) {
        std::ofstream logstream;
        std::string filename = logfile + ".log." + std::to_string(rank);
        printf("Saving log to %s\n", filename.c_str());
        logstream.open(filename);
        logstream << log;
        logstream.close();
    }

    ttor::comms_world_barrier();
    if(test && rank_k == 0) {
        // Send all to 0
        int n_received = 0;
        int n_expected = (rank == 0 ? n * n * n_ranks_1d * n_ranks_1d : 0);
        Eigen::MatrixXd C_test = Eigen::MatrixXd::Zero(N, N);
        auto comm = ttor::make_communicator_world(verb);
        auto am = comm->make_active_msg([&](const ttor::view<double>& A, const int& rank_i_from, const int& rank_j_from, const int& sub_i, const int& sub_j){
            C_test.block(rank_i_from * Nr + sub_i * Nt, rank_j_from * Nr + sub_j * Nt, Nt, Nt) = make_from_view(A, Nt);
            n_received++;
        });
        int rank_i_from = rank_i;
        int rank_j_from = rank_j;
        for(int sub_i = 0; sub_i < n; sub_i++) {
            for(int sub_j = 0; sub_j < n; sub_j++) {
                auto C_view = make_view(&C_ij[sub_i][sub_j]);
                am->send(0, C_view, rank_i_from, rank_j_from, sub_i, sub_j);
            }
        }
        while((!comm->is_done()) || (n_received < n_expected)) {
            comm->progress();
        }
        // Compute reference on 0
        if(rank == 0) {
            Eigen::MatrixXd A_ref = Eigen::MatrixXd::NullaryExpr(N, N, [](int i, int j){return val_global(i,j);});
            Eigen::MatrixXd B_ref = Eigen::MatrixXd::NullaryExpr(N, N, [](int i, int j){return val_global(i,j);});
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

    cxxopts::Options options("3d_gemm", "3D dense gemm using TaskTorrent");
    options.add_options()
        ("help", "Print help")
        ("n_threads", "Number of threads", cxxopts::value<int>()->default_value("1"))
        ("verb", "Verbosity level", cxxopts::value<int>()->default_value("0"))
        ("block_size", "Block size", cxxopts::value<int>()->default_value("8"))
        ("matrix_size", "Total matrix size", cxxopts::value<int>()->default_value("128"))
        ("logfile", "TTOR logging file", cxxopts::value<std::string>()->default_value(""))
        ("test", "Test or not", cxxopts::value<bool>()->default_value("true"));
    auto result = options.parse(argc, argv);

    const int n_threads = result["n_threads"].as<int>();
    const int verb = result["verb"].as<int>();
    const int block_size = result["block_size"].as<int>();
    const int matrix_size = result["matrix_size"].as<int>();
    const bool test = result["test"].as<bool>();
    const std::string logfile = result["logfile"].as<std::string>();

    assert(block_size > 0);
    assert(matrix_size > 0);
    assert(n_threads > 0);
    assert(verb >= 0);

    if (result.count("help")) {
        std::cout << options.help({"", "Group"}) << std::endl;
        exit(0);
    }
    if(ttor::comms_world_rank() == 0) printf("Arguments: block_size %d\nmatrix_size %d\nn_threads %d\nverb %d\ntest %d\nlogfile %s\n", 
        block_size, matrix_size, n_threads, verb, test, logfile.c_str());

    gemm(matrix_size, block_size, n_threads, logfile, verb, test);

    ttor::comms_finalize();
}
