#include "communications.hpp"
#include "runtime.hpp"
#include "util.hpp"
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <fstream>
#include <array>
#include <random>
#include <mutex>
#include <iostream>
#include <map>
#include <memory>

#include <mpi.h>

typedef array<int, 2> int2;
typedef array<int, 3> int3;
typedef array<int, 4> int4;
typedef array<int, 5> int5;
typedef array<int, 6> int6;
typedef array<int, 7> int7;

struct scoped_timer {
  private:
    std::atomic<long long int>* time_us_;
    ttor::timer time_init_;
  public:
    scoped_timer(std::atomic<long long int>* time_us) {
        time_us_ = time_us;
        time_init_= ttor::wctime();
    }
    ~scoped_timer() {
        ttor::timer time_end_ = ttor::wctime();
        *time_us_ += static_cast<long long int>(1e6 * ttor::elapsed(time_init_, time_end_));
    }
};

ttor::view<double> make_view(Eigen::MatrixXd* A) {
    return ttor::view<double>(A->data(), A->size());
}

Eigen::MatrixXd make_from_view(ttor::view<double> A, int nrows) {
    Eigen::MatrixXd Add = Eigen::MatrixXd::Zero(nrows, nrows);
    assert(nrows * nrows == A.size());
    memcpy(Add.data(), A.data(), sizeof(double) * A.size());
    return Add;
}

void copy_from_view(Eigen::MatrixXd* dest, const ttor::view<double> A) {
    assert(dest->size() == A.size());
    memcpy(dest->data(), A.data(), sizeof(double) * A.size());
}

void accumulate(Eigen::MatrixXd* dest, const Eigen::MatrixXd* src) {
    assert(dest->size() == src->size());
    #pragma omp parallel for
    for(int k = 0; k < dest->size(); k++) {
        (*dest)(k) += (*src)(k);
    }
}

std::string to_string(int2 ij) {
    return to_string(ij[0]) + "_" + to_string(ij[1]);
}

std::string to_string(int3 ijk) {
    return to_string(ijk[0]) + "_" + to_string(ijk[1]) + "_" + to_string(ijk[2]);
}

/**
 * Matrix is of size N (global)
 * Each rank works on Nr x Nr
 * Each thread works on Nt x Nt
 * 
 * There are n sub blocks on a given rank
 */
void gemm(const int N, const int Nt, const int n_threads, std::string logfile, const int verb, const bool test)
{
    const int rank = ttor::comm_rank();
    const int n_ranks = ttor::comm_size();
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
    printf("Hello rank %d with 3d-index (%d %d %d) / (%d %d %d) from host %s, N %d, Nr %d, Nt %d, n %d\n", rank, rank_i, rank_j, rank_k, n_ranks_1d, n_ranks_1d, n_ranks_1d, 
        ttor::processor_name().c_str(), N, Nr, Nt, n);
    
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
    std::atomic<long long int> accu_am_us_t(0);

    /**
     * Original and final matrices
     **/
    Eigen::MatrixXd A_ij = Eigen::MatrixXd::Zero(Nr, Nr);
    Eigen::MatrixXd B_ij = Eigen::MatrixXd::Zero(Nr, Nr);
    Eigen::MatrixXd C_ij = Eigen::MatrixXd::Zero(Nr, Nr);

    auto val_global = [](int i, int j) { return static_cast<double>(1 + i + j); };
    if(rank_k == 0) {
        auto val = [&](int i, int j) { return val_global(rank_i * Nr + i, rank_j * Nr + j); };
        A_ij = Eigen::MatrixXd::NullaryExpr(Nr, Nr, val);
        B_ij = Eigen::MatrixXd::NullaryExpr(Nr, Nr, val);
    }

    /** 
     * Workspace
     **/
    Eigen::MatrixXd C_ijk = Eigen::MatrixXd::Zero(Nr, Nr);
    Eigen::MatrixXd A_ijk = Eigen::MatrixXd::Zero(Nr, Nr);
    Eigen::MatrixXd B_ijk = Eigen::MatrixXd::Zero(Nr, Nr);
    MPI_Barrier(MPI_COMM_WORLD);

    /**
     * Initialize the runtime structures
     **/
    ttor::Communicator comm(verb);
    ttor::Threadpool tp(n_threads, &comm, verb, "Wk_Gemm_" + to_string(rank) + "_");

    // send is indexed by int2, which are the sub blocks
    ttor::Taskflow<int2> send_Aij(&tp, verb);  // (i,j,0) sends A_ij to (i,j,j) for all i,j
    ttor::Taskflow<int2> send_Bij(&tp, verb);  // (i,j,0) sends B_ij to (i,j,i) for all i,j
    // send is indexed by int2, which are the sub blocks
    ttor::Taskflow<int2> bcst_Aij(&tp, verb);  // (i,j,j) sends A_ij along j to all (i,*,j) for all i,j
    ttor::Taskflow<int2> bcst_Bij(&tp, verb);  // (i,j,i) sends B_ij along i to all (*,j,i) for all i,j
    // gemm is indexed by int3, which are the sub blocks
    ttor::Taskflow<int3> gemm_Cijk(&tp, verb); // (i,j,k) compute C_ijk = A_ik * B_kj, send for accumulation reduction on (i,j,0)

    ttor::Logger log(1000000);
    if(logfile.size() > 0) {
        tp.set_logger(&log);
    }

    /** 
     * Send
     **/

    auto send_Aij_am = comm.make_active_msg([&](ttor::view<double>& Aij, int& sub_i, int& sub_j) {
        scoped_timer t(&send_am_us_t);
        A_ijk.block(sub_i * Nt, sub_j * Nt, Nt, Nt) = make_from_view(Aij, Nt);
        bcst_Aij.fulfill_promise({sub_i, sub_j});
    });

    auto send_Bij_am = comm.make_active_msg([&](ttor::view<double>& Bij, int& sub_i, int& sub_j) {
        scoped_timer t(&send_am_us_t);
        B_ijk.block(sub_i * Nt, sub_j * Nt, Nt, Nt) = make_from_view(Bij, Nt);
        bcst_Bij.fulfill_promise({sub_i, sub_j});
    });

    // (i,j,0) sends A_ij to (i,j,j) for all i,j
    send_Aij.set_task([&](int2 sub_ij){
        assert(rank_k == 0);
        scoped_timer t(&send_copy_us_t);
        int sub_i = sub_ij[0];
        int sub_j = sub_ij[1];
        Eigen::MatrixXd A_subij = A_ij.block(sub_i * Nt, sub_j * Nt, Nt, Nt);
        ttor::view<double> A_view = make_view(&A_subij);
        int dest = rank_ijk_to_rank(rank_i, rank_j, rank_j);
        send_Aij_am->send(dest, A_view, sub_i, sub_j);
    }).set_indegree([&](int2 ijk) {
        return 1;
    }).set_mapping([&](int2 ijk) {
        return 0;
    }).set_name([&](int2 ijk) { return "send_A_" + to_string(ijk) + "_" + to_string(rank_ijk); });

    // (i,j,0) sends B_ij to (i,j,i) for all i,j
    send_Bij.set_task([&](int2 sub_ij){
        assert(rank_k == 0);
        scoped_timer t(&send_copy_us_t);
        int sub_i = sub_ij[0];
        int sub_j = sub_ij[1];
        Eigen::MatrixXd B_subij = B_ij.block(sub_i * Nt, sub_j * Nt, Nt, Nt);
        ttor::view<double> B_view = make_view(&B_subij);
        int dest = rank_ijk_to_rank(rank_i, rank_j, rank_i);
        send_Bij_am->send(dest, B_view, sub_i, sub_j);
    }).set_indegree([&](int2 ijk) {
        return 1;
    }).set_mapping([&](int2 ijk) {
        return 0;
    }).set_name([&](int2 ijk) { return "send_B_" + to_string(ijk) + "_" + to_string(rank_ijk); });

    /** 
     * Broadcast
     **/

    auto bcst_Aij_am = comm.make_active_msg([&](ttor::view<double>& Aij, int &sub_i, int &sub_j) {
        scoped_timer t(&bcst_am_us_t);
        A_ijk.block(sub_i * Nt, sub_j * Nt, Nt, Nt) = make_from_view(Aij, Nt);
        for(int k = 0; k < n; k++)
            gemm_Cijk.fulfill_promise({sub_i, k, sub_j});
    });

    auto bcst_Bij_am = comm.make_active_msg([&](ttor::view<double>& Bij, int &sub_i, int &sub_j) {
        scoped_timer t(&bcst_am_us_t);
        B_ijk.block(sub_i * Nt, sub_j * Nt, Nt, Nt) = make_from_view(Bij, Nt);
        for(int k = 0; k < n; k++)
            gemm_Cijk.fulfill_promise({k, sub_j, sub_i});
    });

    // (i,j,j) sends A_ij along j to all (i,*,j) for all i,j
    bcst_Aij.set_task([&](int2 sub_ij){
        scoped_timer t(&bcst_copy_us_t);
        assert(rank_j == rank_k);
        int sub_i = sub_ij[0];
        int sub_j = sub_ij[1];
        Eigen::MatrixXd A_subij = A_ijk.block(sub_i * Nt, sub_j * Nt, Nt, Nt);
        ttor::view<double> A_view = make_view(&A_subij);
        for(int k = 0; k < n_ranks_1d; k++) {
            int dest = rank_ijk_to_rank(rank_i, k, rank_j);
            bcst_Aij_am->send(dest, A_view, sub_i, sub_j);
        }
    }).set_indegree([&](int2 ij) {
        return 1;
    }).set_mapping([&](int2 ij) {
        return 0;
    }).set_name([&](int2 ij) { return "bcast_A_" + to_string(ij) + "_" + to_string(rank_ijk); });

    // (i,j,i) sends B_ij along i to all (*,j,i) for all i,j
    bcst_Bij.set_task([&](int2 sub_ij){
        scoped_timer t(&bcst_copy_us_t);
        assert(rank_i == rank_k);
        int sub_i = sub_ij[0];
        int sub_j = sub_ij[1];
        Eigen::MatrixXd B_subij = B_ijk.block(sub_i * Nt, sub_j * Nt, Nt, Nt);
        ttor::view<double> B_view = make_view(&B_subij);
        for(int k = 0; k < n_ranks_1d; k++) {
            int dest = rank_ijk_to_rank(k, rank_j, rank_i);
            bcst_Bij_am->send(dest, B_view, sub_i, sub_j);
        }
    }).set_indegree([&](int2 ij) {
        return 1;
    }).set_mapping([&](int2 ij) {
        return 0;
    }).set_name([&](int2 ij) { return "bcast_B_" + to_string(ij) + "_" + to_string(rank_ijk); });

    /** 
     * GEMM
     **/

    auto accu_Cijk_am = comm.make_active_msg([&](ttor::view<double>& Cijk, int &sub_i, int &sub_j) {
        scoped_timer t(&accu_am_us_t);
        C_ij.block(sub_i * Nt, sub_j * Nt, Nt, Nt) += make_from_view(Cijk, Nt);
    });

    // (i,j,k) compute C_ijk = A_ik * B_kj
    gemm_Cijk.set_task([&](int3 sub_ijk){
        int sub_i = sub_ijk[0];
        int sub_j = sub_ijk[1];
        int sub_k = sub_ijk[2];
        {
            scoped_timer t(&gemm_us_t);
            C_ijk.block(sub_i * Nt, sub_j * Nt, Nt, Nt).noalias() +=
                A_ijk.block(sub_i * Nt, sub_k * Nt, Nt, Nt) *
                B_ijk.block(sub_k * Nt, sub_j * Nt, Nt, Nt);
        }
        scoped_timer t(&gemm_copy_us_t);
        if(sub_ijk[2] < n-1) {
            gemm_Cijk.fulfill_promise({sub_i, sub_j, sub_k+1});
        } else {
            Eigen::MatrixXd C_ijk_tmp = C_ijk.block(sub_i * Nt, sub_j * Nt, Nt, Nt);
            auto C_ijk_view = make_view(&C_ijk_tmp);
            int dest = rank_ijk_to_rank(rank_i, rank_j, 0);
            int k = rank_k;
            accu_Cijk_am->send(dest, C_ijk_view, sub_i, sub_j);
        }
    }).set_indegree([&](int3 ijk) {
        return ijk[2] == 0 ? 2 : 3; // 2 A_ik and B_kj blocks, + previous gemm
    }).set_mapping([&](int3 ijk) {
        return 0;
    }).set_name([&](int3 ijk) { return "gemm_C_" + to_string(ijk) + "_" + to_string(rank_ijk); });

    printf("Starting 3D Gemm...\n");
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
    ttor::timer t1 = ttor::wctime();
    printf("Done\n");
    printf("total_time,%e\n", ttor::elapsed(t0, t1));
    printf("send_copy_us_t,%e\n",send_copy_us_t.load() * 1e-6);
    printf("send_am_us_t,%e\n",send_am_us_t.load() * 1e-6);
    printf("bcst_copy_us_t,%e\n",bcst_copy_us_t.load() * 1e-6);
    printf("bcst_am_us_t,%e\n",bcst_am_us_t.load() * 1e-6);
    printf("gemm_us_t,%e\n",gemm_us_t.load() * 1e-6);
    printf("gemm_copy_us_t,%e\n",gemm_copy_us_t.load() * 1e-6);
    printf("accu_am_us_t,%e\n",accu_am_us_t.load() * 1e-6);

    if(logfile.size() > 0) {
        std::ofstream logstream;
        std::string filename = logfile + ".log." + to_string(rank);
        printf("Saving log to %s\n", filename.c_str());
        logstream.open(filename);
        logstream << log;
        logstream.close();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if(test && rank_k == 0) {
        // Send all to 0
        int n_received = 0;
        int n_expected = (rank == 0 ? n * n * n_ranks_1d * n_ranks_1d : 0);
        Eigen::MatrixXd C_test = Eigen::MatrixXd::Zero(N, N);
        ttor::Communicator comm(verb);
        auto am = comm.make_active_msg([&](ttor::view<double>& A, int& rank_i_from, int& rank_j_from, int& sub_i, int& sub_j){
            C_test.block(rank_i_from * Nr + sub_i * Nt, rank_j_from * Nr + sub_j * Nt, Nt, Nt) = make_from_view(A, Nt);
            n_received++;
        });
        int rank_i_from = rank_i;
        int rank_j_from = rank_j;
        for(int sub_i = 0; sub_i < n; sub_i++) {
            for(int sub_j = 0; sub_j < n; sub_j++) {
                Eigen::MatrixXd C_ij_tmp = C_ij.block(sub_i * Nt, sub_j * Nt, Nt, Nt);
                auto C_view = make_view(&C_ij_tmp);
                am->send(0, C_view, rank_i_from, rank_j_from, sub_i, sub_j);
            }
        }
        while((!comm.is_done()) || (n_received < n_expected)) {
            comm.progress();
        }
        // Compute reference on 0
        if(rank == 0) {
            Eigen::MatrixXd A_ref = Eigen::MatrixXd::NullaryExpr(N, N, val_global);
            Eigen::MatrixXd B_ref = Eigen::MatrixXd::NullaryExpr(N, N, val_global);
            ttor::timer t0 = ttor::wctime();
            Eigen::MatrixXd C_ref = A_ref * B_ref;
            ttor::timer t1 = ttor::wctime();
            double error = (C_ref - C_test).norm() / C_ref.norm();
            printf("\n==> GEMM error %e\n\n", error);
            printf("Reference code took %e\n", ttor::elapsed(t0, t1));
            assert(error <= 1e-12);
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

    if (argc >= 5) {
        logfile = argv[4];
    }

    if (argc >= 6) {
        verb = atoi(argv[5]);
        assert(verb >= 0);
    }

    if (argc >= 7) {
        test = static_cast<bool>(atoi(argv[6]));
    }

    if(ttor::comm_rank() == 0) printf("Usage: ./3d_gemm N Nt n_threads logfile verb test\n");
    if(ttor::comm_rank() == 0) printf("Arguments: N (global matrix size) %d, Nt (smallest block size) %d, n_threads %d, logfile %s, verb %d, test %d\n", N, Nt, n_threads, logfile.c_str(), verb, test);

    gemm(N, Nt, n_threads, logfile, verb, test);

    MPI_Finalize();
}
