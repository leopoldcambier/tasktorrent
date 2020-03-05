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

void gemm(const int matrix_size, const int verb, const bool test)
{
    const int n_threads = 1; // Shared memory parallelism is in BLAS 
    const int rank = ttor::comm_rank();
    const int n_ranks = ttor::comm_size();
    const int n_ranks_1d = static_cast<int>(round(pow(n_ranks, 1.0/3.0)));
    assert(n_ranks_1d * n_ranks_1d * n_ranks_1d == n_ranks);
    const int rank_i = rank % n_ranks_1d;
    const int rank_j = (rank / n_ranks_1d) % n_ranks_1d;
    const int rank_k = rank / (n_ranks_1d * n_ranks_1d);
    const int3 rank_ijk = {rank_i, rank_j, rank_k};
    const int block_size = matrix_size / n_ranks_1d;
    assert(block_size * n_ranks_1d == matrix_size);
    printf("Hello rank %d with 3d-index (%d %d %d) / (%d %d %d) from host %s\n", rank, rank_i, rank_j, rank_k, n_ranks_1d, n_ranks_1d, n_ranks_1d, ttor::processor_name().c_str());
    
    auto rank_ijk_to_rank = [n_ranks_1d](int rank_i, int rank_j, int rank_k) {
        return rank_k * n_ranks_1d * n_ranks_1d + rank_j * n_ranks_1d + rank_i;
    };
    assert(rank_ijk_to_rank(rank_i, rank_j, rank_k) == rank);

    /**
     * Record timings
     **/
    std::atomic<long long int> send_copy_us_t(0);
    std::atomic<long long int> bcst_copy_us_t(0);
    std::atomic<long long int> accu_copy_us_t(0);
    std::atomic<long long int> gemm_us_t(0);
    std::atomic<long long int> accu_us_t(0);

    /**
     *  Initializes the matrices
     **/
    Eigen::MatrixXd A_ij = Eigen::MatrixXd::Zero(block_size, block_size);
    Eigen::MatrixXd B_ij = Eigen::MatrixXd::Zero(block_size, block_size);
    Eigen::MatrixXd C_ij = Eigen::MatrixXd::Zero(block_size, block_size);
    auto val_global = [](int i, int j) { return static_cast<double>(1 + i + j); };
    if(rank_k == 0) {
        auto val = [&](int i, int j) { return val_global(rank_i * block_size + i, rank_j * block_size + j); };
        A_ij = Eigen::MatrixXd::NullaryExpr(block_size, block_size, val);
        B_ij = Eigen::MatrixXd::NullaryExpr(block_size, block_size, val);
    }
    Eigen::MatrixXd C_ijk = Eigen::MatrixXd::Zero(block_size, block_size);
    vector<Eigen::MatrixXd> C_ijks(n_ranks_1d, Eigen::MatrixXd::Zero(block_size, block_size));

    /**
     * Initialize the runtime structures
     **/
    ttor::Communicator comm(verb);
    ttor::Threadpool tp(n_threads, &comm, verb, "Wk_Gemm_" + to_string(rank) + "_");
    ttor::Taskflow<int> send_Aij(&tp, verb);  // (i,j,0) sends A_ij to (i,j,j) for all i,j
    ttor::Taskflow<int> send_Bij(&tp, verb);  // (i,j,0) sends B_ij to (i,j,i) for all i,j
    ttor::Taskflow<int> bcst_Aij(&tp, verb);  // (i,j,j) sends A_ij along j to all (i,*,j) for all i,j
    ttor::Taskflow<int> bcst_Bij(&tp, verb);  // (i,j,i) sends B_ij along i to all (*,j,i) for all i,j
    ttor::Taskflow<int> gemm_Cijk(&tp, verb); // (i,j,k) compute C_ijk = A_ik * B_kj, send for reduction on (i,j,0)
    ttor::Taskflow<int> accu_Cijk(&tp, verb); // (i,j,0) accumulates C_ijk

    /** 
     * Send
     **/

    auto send_Aij_am = comm.make_active_msg([&](ttor::view<double>& Aij) {
        ttor::timer t0 = ttor::wctime();
        copy_from_view(&A_ij, Aij);
        ttor::timer t1 = ttor::wctime();
        send_copy_us_t += 1e6 * ttor::elapsed(t0, t1);
        bcst_Aij.fulfill_promise(0);
    });

    auto send_Bij_am = comm.make_active_msg([&](ttor::view<double>& Bij) {
        ttor::timer t0 = ttor::wctime();
        copy_from_view(&B_ij, Bij);
        ttor::timer t1 = ttor::wctime();
        send_copy_us_t += 1e6 * ttor::elapsed(t0, t1);
        bcst_Bij.fulfill_promise(0);
    });

    // (i,j,0) sends A_ij to (i,j,j) for all i,j
    send_Aij.set_task([&](int ijk){
        assert(rank_k == 0);
        ttor::view<double> A_view = make_view(&A_ij);
        int dest = rank_ijk_to_rank(rank_i, rank_j, rank_j);
        if(dest != rank) {
            send_Aij_am->send(dest, A_view);
        } else {
            bcst_Aij.fulfill_promise(0);
        }
    }).set_indegree([&](int ijk) {
        return 1;
    }).set_mapping([&](int ijk) {
        return 0;
    }).set_name([&](int ijk) { return "send_A_" + to_string(rank_ijk); });

    // (i,j,0) sends B_ij to (i,j,i) for all i,j
    send_Bij.set_task([&](int ijk){
        assert(rank_k == 0);
        ttor::view<double> B_view = make_view(&B_ij);
        int dest = rank_ijk_to_rank(rank_i, rank_j, rank_i);
        if(dest != rank) {
            send_Bij_am->send(dest, B_view);
        } else {
            bcst_Bij.fulfill_promise(0);
        }
    }).set_indegree([&](int ijk) {
        return 1;
    }).set_mapping([&](int ijk) {
        return 0;
    }).set_name([&](int ijk) { return "send_B_" + to_string(rank_ijk); });

    /** 
     * Broadcast
     **/

    auto bcst_Aij_am = comm.make_active_msg([&](ttor::view<double>& Aij) {
        ttor::timer t0 = ttor::wctime();
        copy_from_view(&A_ij, Aij);
        gemm_Cijk.fulfill_promise(0);
        ttor::timer t1 = ttor::wctime();
        bcst_copy_us_t += 1e6 * ttor::elapsed(t0, t1);
    });

    auto bcst_Bij_am = comm.make_active_msg([&](ttor::view<double>& Bij) {
        ttor::timer t0 = ttor::wctime();
        copy_from_view(&B_ij, Bij);
        gemm_Cijk.fulfill_promise(0);
        ttor::timer t1 = ttor::wctime();
        bcst_copy_us_t += 1e6 * ttor::elapsed(t0, t1);
    });

    // (i,j,j) sends A_ij along j to all (i,*,j) for all i,j
    bcst_Aij.set_task([&](int ijk){
        assert(rank_j == rank_k);
        ttor::view<double> A_view = make_view(&A_ij);
        for(int k = 0; k < n_ranks_1d; k++) {
            int dest = rank_ijk_to_rank(rank_i, k, rank_j);
            if(dest != rank) {
                bcst_Aij_am->send(dest, A_view);
            } else {
                gemm_Cijk.fulfill_promise(0);
            }
        }
    }).set_indegree([&](int ij) {
        return 1;
    }).set_mapping([&](int ij) {
        return 0;
    }).set_name([&](int ijk) { return "bcast_A_" + to_string(rank_ijk); });

    // (i,j,i) sends B_ij along i to all (*,j,i) for all i,j
    bcst_Bij.set_task([&](int ij){
        assert(rank_i == rank_k);
        ttor::view<double> B_view = make_view(&B_ij);
        for(int k = 0; k < n_ranks_1d; k++) {
            int dest = rank_ijk_to_rank(k, rank_j, rank_i);
            if(dest != rank) {
                bcst_Bij_am->send(dest, B_view);
            } else {
                gemm_Cijk.fulfill_promise(0);
            }
        }
    }).set_indegree([&](int ijk) {
        return 1;
    }).set_mapping([&](int ijk) {
        return 0;
    }).set_name([&](int ijk) { return "bcast_B_" + to_string(rank_ijk); });

    /** 
     * GEMM
     **/

    auto accu_Cijk_am = comm.make_active_msg([&](ttor::view<double>& Cijk, int& k) {
        ttor::timer t0 = ttor::wctime();
        copy_from_view(&C_ijks[k], Cijk);
        accu_Cijk.fulfill_promise(k);
        ttor::timer t1 = ttor::wctime();
        accu_copy_us_t += 1e6 * ttor::elapsed(t0, t1);
    });

    // (i,j,k) compute C_ijk = A_ik * B_kj
    gemm_Cijk.set_task([&](int ijk){
        // A_ij is actually A_(rank_i, rank_k) now
        // B_ij is actually B_(rank_k, rank_j) now
        ttor::timer t0 = ttor::wctime();
        C_ijk.noalias() += A_ij * B_ij;
        ttor::timer t1 = ttor::wctime();
        gemm_us_t += 1e6 * ttor::elapsed(t0, t1);
        auto C_ijk_view = make_view(&C_ijk);
        int dest = rank_ijk_to_rank(rank_i, rank_j, 0);
        int k = rank_k;
        accu_Cijk_am->send(dest, C_ijk_view, k);
    }).set_indegree([&](int ij) {
        return 2;
    }).set_mapping([&](int ij) {
        return 0;
    }).set_name([&](int ijk) { return "gemm_C_" + to_string(rank_ijk); });

    /** 
     * ACCUMULATE
     **/
    accu_Cijk.set_task([&](int k){
        ttor::timer t0 = ttor::wctime();
        accumulate(&C_ij, &C_ijks[k]);
        ttor::timer t1 = ttor::wctime();
        accu_us_t += 1e6 * ttor::elapsed(t0, t1);
    }).set_indegree([&](int ij) {
        return 1;
    }).set_mapping([&](int ij) {
        return 0;
    }).set_binding([&](int k) {
        return true;
    }).set_name([&](int ijk) { return "accu_C_" + to_string(rank_ijk); });

    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0) printf("Starting 3D Gemm...\n");
    ttor::timer t0 = ttor::wctime();
    if(rank_k == 0) {
        send_Aij.fulfill_promise(0);
        send_Bij.fulfill_promise(0);
    }
    tp.join();
    MPI_Barrier(MPI_COMM_WORLD);
    ttor::timer t1 = ttor::wctime();
    if(rank == 0) printf("Done\n");
    if(rank == 0) printf("Elapsed time: %e\n", ttor::elapsed(t0, t1));

    printf("gemm,%e\n", gemm_us_t.load() * 1e-6);
    printf("accu,%e\n", accu_us_t.load() * 1e-6);
    printf("send_copy,%e\n", send_copy_us_t.load() * 1e-6);
    printf("bcst_copy,%e\n", bcst_copy_us_t.load() * 1e-6);
    printf("accu_copy,%e\n", accu_copy_us_t.load() * 1e-6);

    if(test && rank_k == 0) {
        // Send all to 0
        int n_received = 0;
        int n_expected = (rank == 0 ? n_ranks_1d * n_ranks_1d : 0);
        Eigen::MatrixXd C_test = Eigen::MatrixXd::Zero(matrix_size, matrix_size);
        ttor::Communicator comm(verb);
        auto am = comm.make_active_msg([&](ttor::view<double>& A, int& rank_i_from, int& rank_j_from){
            C_test.block(rank_i_from * block_size, rank_j_from * block_size, block_size, block_size) = make_from_view(A, block_size);
            n_received++;
        });
        auto C_view = make_view(&C_ij);
        int rank_i_from = rank_i;
        int rank_j_from = rank_j;
        am->send(0, C_view, rank_i_from, rank_j_from);
        while((!comm.is_done()) || (n_received < n_expected)) {
            comm.progress();
        }
        // Compute reference on 0
        if(rank == 0) {
            Eigen::MatrixXd A_ref = Eigen::MatrixXd::NullaryExpr(matrix_size, matrix_size, val_global);
            Eigen::MatrixXd B_ref = Eigen::MatrixXd::NullaryExpr(matrix_size, matrix_size, val_global);
            ttor::timer t0 = ttor::wctime();
            Eigen::MatrixXd C_ref = A_ref * B_ref;
            ttor::timer t1 = ttor::wctime();
            double error = (C_ref - C_test).norm() / C_ref.norm();
            printf("GEMM error %e\n", error);
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

    int matrix_size = 128;
    int verb = 0;
    bool test = true;

    if (argc >= 2)
    {
        matrix_size = atoi(argv[1]);
        assert(matrix_size > 0);
    }
    
    if (argc >= 3) {
        verb = atoi(argv[2]);
        assert(verb >= 0);
    }

    if (argc >= 4) {
        test = static_cast<bool>(atoi(argv[3]));
    }

    if(ttor::comm_rank() == 0) printf("Usage: ./3d_gemm matrix_size verb test\n");
    if(ttor::comm_rank() == 0) printf("Arguments: matrix_size (global matrix size) %d, verb %d, test %d\n", matrix_size, verb, test);

    gemm(matrix_size, verb, test);

    MPI_Finalize();
}
