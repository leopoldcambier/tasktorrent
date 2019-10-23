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

#include <gtest/gtest.h>
#include <mpi.h>

using namespace std;
using namespace Eigen;
using namespace ttor;

typedef array<int, 2> int2;
typedef array<int, 3> int3;

int VERB = 1;
int n_threads_ = 4;
int n_ = 100;
int N_ = 20;

void cholesky(int n_threads, int n, int N)
{
    // MPI info
    const int rank = comm_rank();
    const int n_ranks = comm_size();

    // Form the matrix : let every node have a copy of A for now
    std::default_random_engine gen;
    std::uniform_int_distribution<> dist(-1, 1);
    auto rnd = [&](int i, int j) { return i == j ? 4 : i+j; };

    MatrixXd A_ = MatrixXd::NullaryExpr(N * n, N * n, rnd);

    MatrixXd A = A_ * A_.transpose();
    A = A.triangularView<Lower>();
    MatrixXd Aref = A;

    // Initialize the communicator structure
    Communicator comm(VERB);

    // Threadpool
    Threadpool tp(n_threads, &comm, VERB);
    Taskflow<int> potf_tf(&tp, VERB);
    Taskflow<int2> trsm_tf(&tp, VERB);
    Taskflow<int3> gemm_tf(&tp, VERB);

    // Log
    DepsLogger dlog(1000000);
    Logger log(1000000);
    tp.set_logger(&log);

    // Active messages
    auto am_trsm = comm.make_active_msg(
        [&](view<double> &vkk, int2& ki) {
        	int k = ki[0];
        	A.block(k * n, k * n, n, n) = Map<MatrixXd> (vkk.data(), n,n);
          	trsm_tf.fulfill_promise(ki);
        });

    auto am_gemm = comm.make_active_msg(
        [&](view<double> &vik, int2& ki, int3& kli) {
        	int k = ki[0];
        	int i = ki[1];
        	A.block(i * n, k * n, n, n) = Map<MatrixXd> (vik.data(), n,n);
          	gemm_tf.fulfill_promise(kli);
          	
        });

    // potf 
    potf_tf.set_mapping([&](int k) {
               return (k % n_threads);
           })
        .set_indegree([](int k) {
            return 1;
        })
        .set_task([&](int k) {
            LLT<MatrixXd> llt;
            llt.compute(A.block(k * n, k * n, n, n).selfadjointView<Lower>());
            A.block(k * n, k * n, n, n) = llt.matrixL();
           
        })
        .set_fulfill([&](int k) {
            for (int i = k + 1; i < N; i++)
            {
            	if (i%n_ranks == rank){
            		trsm_tf.fulfill_promise({k, i});
                	dlog.add_event(DepsEvent(potf_tf.name(k), trsm_tf.name({k, i})));
            	}
            	else {
            		MatrixXd Akk = A.block(k * n, k * n, n, n);    
            		int2 ki = {k,i};
            		auto vkk = view<double>(Akk.data(), Akk.size());
            		am_trsm->send(i%n_ranks, vkk ,ki);
            		dlog.add_event(DepsEvent(potf_tf.name(k), trsm_tf.name(ki)));
            	}
                
            }
        })
        .set_name([](int k) {
            return "potf_" + to_string(k);
        })
        .set_priority([](int k) {
            return 3.0;
        });

    // trsm
    trsm_tf.set_mapping([&](int2 ki) {
               return ((ki[0] + ki[1] * N) % n_threads);
           })
        .set_indegree([](int2 ki) {
            return (ki[0] == 0 ? 0 : 1) + 1;
        })
        .set_task([&](int2 ki) {
            int k = ki[0];
            int i = ki[1];
            auto L = A.block(k * n, k * n, n, n).triangularView<Lower>().transpose();
            A.block(i * n, k * n, n, n) = L.solve<OnTheRight>(A.block(i * n, k * n, n, n));
            
        })
        .set_fulfill([&](int2 ki) {
            int k = ki[0];
            int i = ki[1];
            for (int l = k + 1; l <= i; l++)
            {
                gemm_tf.fulfill_promise({k, i, l}); // happens on the same node
                dlog.add_event(DepsEvent(trsm_tf.name(ki), gemm_tf.name({k, i, l})));
            }
            for (int l = i + 1; l < N; l++)
            {
            	if (l%n_ranks == rank){
            		gemm_tf.fulfill_promise({k, l, i});
                	dlog.add_event(DepsEvent(trsm_tf.name(ki), gemm_tf.name({k, l, i})));
            	}
            	else{
            		MatrixXd Aik = A.block(i * n, k * n, n, n);
            		int3 kli = {k,l,i};
            		auto vik = view<double>(Aik.data(), Aik.size());
            		am_gemm->send(l%n_ranks, vik, ki,kli);
            		dlog.add_event(DepsEvent(trsm_tf.name(ki), gemm_tf.name(kli)));
            	}    
            }
        })
        .set_name([](int2 ki) {
            return "trsm_" + to_string(ki[0]) + "_" + to_string(ki[1]);
        })
        .set_priority([](int2 k) {
            return 2.0;
        });

    // gemm
    gemm_tf.set_mapping([&](int3 kij) {
               return ((kij[0] + kij[1] * N + kij[2] * N * N) % n_threads);
           })
        .set_indegree([](int3 kij) {
            int k = kij[0];
            int i = kij[1];
            int j = kij[2];
            return (k == 0 ? 0 : 1) + (i == j ? 1 : 2);
        })
        .set_task([&](int3 kij) {
            int k = kij[0];
            int i = kij[1];
            int j = kij[2];
            A.block(i * n, j * n, n, n) -= A.block(i * n, k * n, n, n) * A.block(j * n, k * n, n, n).transpose();
            if (i == j)
                A.block(i * n, j * n, n, n) = A.block(i * n, j * n, n, n).triangularView<Lower>();
            ASSERT_TRUE(k < N - 1);
        })
        .set_fulfill([&](int3 kij) {
            int k = kij[0];
            int i = kij[1];
            int j = kij[2];
            if (k + 1 == i && k + 1 == j)
            {
                potf_tf.fulfill_promise(k + 1); // same node
                dlog.add_event(DepsEvent(gemm_tf.name(kij), potf_tf.name(k + 1)));
            }
            else if (k + 1 == j)
            {
                trsm_tf.fulfill_promise({k + 1, i}); // same node
                dlog.add_event(DepsEvent(gemm_tf.name(kij), trsm_tf.name({k + 1, i})));
            }
            else
            {
                gemm_tf.fulfill_promise({k + 1, i, j}); // same node
                dlog.add_event(DepsEvent(gemm_tf.name(kij), gemm_tf.name({k + 1, i, j})));
            }
        })
        .set_name([](int3 kij) {
            return "gemm_" + to_string(kij[0]) + "_" + to_string(kij[1]) + "_" + to_string(kij[2]);
        })
        .set_priority([](int3 k) {
            return 1.0;
        });

        timer t0 = wctime();
        if (rank == 0){
            potf_tf.fulfill_promise(0);
        }
    	tp.join();
    	timer t1 = wctime();

    	std::ofstream logfile;
    	string filename = "cholesky_"+ to_string(rank)+".log";
    	logfile.open(filename);
    	logfile << log;
    	logfile.close();

    	std::ofstream depsfile;
    	string dfilename = "cholesky_"+ to_string(rank)+".dot";
    	depsfile.open(dfilename);
    	depsfile << dlog;
    	depsfile.close();
}

TEST(cholesky, one)
{
    int n_threads = n_threads_;
  	int n = n_;
  	int N = N_;
    cholesky(n_threads, n, N);
}


int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    int req = MPI_THREAD_FUNNELED;
    int prov = -1;

    MPI_Init_thread(NULL, NULL, req, &prov);

    assert(prov == req);


    if (argc >= 2)
    {
        n_threads_ = atoi(argv[1]);
    }

    if (argc >= 3)
    {
        n_ = atoi(argv[2]);
    }

    if (argc >= 4)
    {
        N_ = atoi(argv[3]);
    }

    if (argc >= 5)
    {
        VERB = atoi(argv[4]);
    }

    const int return_flag = RUN_ALL_TESTS();

    MPI_Finalize();

    return return_flag;
}
