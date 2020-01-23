#include "communications.hpp"
#include "runtime.hpp"
#include "util.hpp"
#include <cblas.h>
#include <lapacke.h>
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

using namespace std;
using namespace Eigen;
using namespace ttor;

using namespace std;
using namespace ttor;

typedef array<int, 2> int2;
typedef array<int, 3> int3;

void cholesky(int n_threads, int verb, int n, int nb)
{
    const int rank = comm_rank();
    const int n_ranks = comm_size();
    std::atomic<long long int> potrf_us_t(0);
    std::atomic<long long int> trsm_us_t(0);
    std::atomic<long long int> gemm_us_t(0);
    std::atomic<long long int> accu_us_t(0);

    // Initializes the matrix
    auto val = [&](int i, int j) { return 1/(float)((i-j)*(i-j)+1); };
    MatrixXd A;
    A = MatrixXd::NullaryExpr(n*nb,n*nb,val);
    MatrixXd L = A;
    vector<unique_ptr<MatrixXd>> blocks(nb*nb);
    for (int ii=0; ii<nb; ii++) {
        for (int jj=0; jj<nb; jj++) {
            blocks[ii+jj*nb]=make_unique<MatrixXd>(n,n);
            *blocks[ii+jj*nb]=L.block(ii*n,jj*n,n,n);
        }
    }

    // Holds the temporary matrices result of gemm to be accumulated by accu
    // Each block holds data to be accumulated into a given block[ii+jj*nb]
    struct acc_data {
        std::map<int, std::unique_ptr<MatrixXd>> to_accumulate; // to_accumulate[k] holds matrix result of gemm(k,i,j)
        std::mutex mtx; // Protects that map
    };
    std::vector<acc_data> gemm_results(nb*nb); // gemm_results[ii+jj*nb] holds the data to be accumulated into blocks[ii+jj*nb]

    // Map tasks to rank
    auto block_2_rank = [&](int i, int j) {
        return (i*nb+j)%n_ranks;
    };

    // Initialize the communicator structure
    Communicator comm(verb);

    // Initialize the runtime structures
    Threadpool tp(n_threads, &comm, verb, "Wk_Chol_" + to_string(rank) + "_");
    Taskflow<int>  potrf(&tp, verb);
    Taskflow<int2> trsm(&tp, verb);
    Taskflow<int3> gemm(&tp, verb);
    Taskflow<int3> accu(&tp, verb);

    Logger log(1000000);
    tp.set_logger(&log);
    comm.set_logger(&log);

    // Send a potrf'ed pivot A(k,k) and trigger trsms below requiring A(k,k)
    auto am_trsm = comm.make_active_msg( 
            [&](view<double> &Ljj, int& j, view<int>& is) {
                *blocks[j+j*nb] = Map<MatrixXd>(Ljj.data(), n, n);
                for(auto& i: is) {
                    trsm.fulfill_promise({i,j}, 5.0);
                }
            });

    /**
     * j is the pivot's position at A(j,j)
     */
    potrf.set_task([&](int j) { // A[j,j] -> A[j,j]
            timer t_ = wctime();
            LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', n, blocks[j+j*nb]->data(), n);
            timer t__ = wctime();
            potrf_us_t += 1e6 * elapsed(t_, t__);
        })
        .set_fulfill([&](int j) { // Triggers all trsms on rows i > j, A[i,j]
            vector<vector<int>> fulfill(n_ranks);
            for (int i = j+1; i<nb; i++) {
                fulfill[block_2_rank(i,j)].push_back(i);
            }
            for (int r = 0; r < n_ranks; r++) { // Looping through all outgoing dependency edges
                if (rank == r) {
                    for (auto& i: fulfill[r]) {
                        trsm.fulfill_promise({i,j}, 5.0);
                    }
                } else {
                    auto Ljjv = view<double>(blocks[j+j*nb]->data(), n*n);
                    auto isv = view<int>(fulfill[r].data(), fulfill[r].size());
                    am_trsm->send(r, Ljjv, j, isv);
                }

            }
        })
        .set_indegree([&](int j) {
            return j == 0 ? 1 : j; // Need j accumulations into (j,j) to trigger the potf
        })
        .set_mapping([&](int j) {
            return (j % n_threads);
        })
        .set_name([&](int j) { // This is just for debugging and profiling
            return "POTRF_" + to_string(j) + "_" + to_string(rank);
        });

    // Sends a panel (trsm'ed block A(i,j)) and trigger gemms requiring A(i,j)
    auto am_gemm = comm.make_active_msg(
        [&](view<double> &Lij, int& i, int& j, view<int2>& ijs) {
            *blocks[i+j*nb] = Map<MatrixXd>(Lij.data(), n, n);
            for(auto& ij: ijs) {
                gemm.fulfill_promise({j,ij[0],ij[1]}, 5.0);
            }
        });

    /**
     * ij is (Row, Col) of the block in the matrix at A(i,j)
     **/
    trsm.set_task([&](int2 ij) { // A[j,j] & A[i,j] -> A[i,j]
            int i=ij[0]; 
            int j=ij[1]; 
            assert(i > j);
            timer t_ = wctime();
            cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit, n, n, 1.0, blocks[j+j*nb]->data(),n, blocks[i+j*nb]->data(), n);
            timer t__ = wctime();
            trsm_us_t += 1e6 * elapsed(t_, t__);
        })
        .set_fulfill([&](int2 ij) {
            int i=ij[0];
            int j=ij[1];
            assert(i > j);
            vector<vector<int2>> fulfill(n_ranks);
            for (int k = j+1; k < nb; k++) {
                int ii = std::max(i,k);
                int jj = std::min(i,k);
                fulfill[block_2_rank(ii,jj)].push_back({ii,jj});
            }
            for (int r = 0; r < n_ranks; r++)   // Looping through all outgoing dependency edges
            {
                if (r == rank) {
                    for (auto& ij : fulfill[r]) {
                        gemm.fulfill_promise({j,ij[0],ij[1]}, 5.0);
                    }
                }
                else {
                    auto Lijv = view<double>(blocks[i+j*nb]->data(), n*n);
                    auto ijsv = view<int2>(fulfill[r].data(), fulfill[r].size());
                    am_gemm->send(r, Lijv, i, j, ijsv);
                }
            }
        })
        .set_indegree([&](int2 ij) {
            return 1 + ij[1]; // Potrf above and all gemms before
        })
        .set_mapping([&](int2 ij) {
            return ((ij[0]*nb+ij[1]) % n_threads);
        })
        .set_name([&](int2 ij) { // This is just for debugging and profiling
            return "TRSM_" + to_string(ij[0]) + "_" + to_string(ij[1]) + "_" +to_string(rank);
        });

    /**
     * k is the step (the pivot's position), ij are Row and Column, at A(i,j)
     **/
    gemm.set_task([&](int3 kij) {
            int k=kij[0];
            int i=kij[1];
            int j=kij[2];
            assert(j <= i);
            assert(k < j);
            std::unique_ptr<MatrixXd> Atmp = make_unique<MatrixXd>(n, n); // The matrix is allocated with garbage. The 0 in the BLAS call make sure its overwritten by 0's before doing any math
            timer t_ = wctime();
            if (i == j) {
                cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans, n, n, -1.0, blocks[i+k*nb]->data(), n, 0.0, Atmp->data(), n);
            } else {
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, n, n, n, -1.0,blocks[i+k*nb]->data(), n, blocks[j+k*nb]->data(), n, 0.0, Atmp->data(), n);
            }
            timer t__ = wctime();
            gemm_us_t += 1e6 * elapsed(t_, t__);
            {
                lock_guard<mutex> lock(gemm_results[i+j*nb].mtx);
                gemm_results[i+j*nb].to_accumulate[k] = move(Atmp);
            }
        })
        .set_fulfill([&](int3 kij) {
            accu.fulfill_promise(kij, 5.0);
        })
        .set_indegree([&](int3 kij) {
            return kij[1] == kij[2] ? 1 : 2; // Either one potf or two trsms
        })
        .set_mapping([&](int3 kij) {
            return ((kij[0]*nb*nb+kij[1]+kij[2]*nb) % n_threads);
        })
        .set_name([&](int3 kij) { // This is just for debugging and profiling
            return "GEMM_" + to_string(kij[0]) + "_" + to_string(kij[1])+"_"+to_string(kij[2])+"_"+to_string(comm_rank());
        });

    /**
     * k is the step (the pivot's position), ij are Row and Column, at A(i,j)
     **/
    accu.set_task([&](int3 kij) {
            int k=kij[0]; // Step (gemm's pivot)
            int i=kij[1]; // Row
            int j=kij[2]; // Col
            assert(j <= i);
            assert(k < j);
            std::unique_ptr<MatrixXd> Atmp;
            {
                lock_guard<mutex> lock(gemm_results[i+j*nb].mtx);
                Atmp = move(gemm_results[i+j*nb].to_accumulate[k]);
                gemm_results[i+j*nb].to_accumulate.erase(k);
            }
            timer t_ = wctime();
            *blocks[i+j*nb] += (*Atmp);
            timer t__ = wctime();
            accu_us_t += 1e6 * elapsed(t_, t__);
        })
        .set_fulfill([&](int3 kij) {
            int k=kij[0];
            int i=kij[1];
            int j=kij[2];
            assert(j <= i);
            assert(k < j);
            if(i == j) {
                potrf.fulfill_promise(i, 5.0);
            } else {
                trsm.fulfill_promise({i,j}, 5.0);
            }
        })
        .set_indegree([&](int3 kij) {
            return 1;
        })
        .set_mapping([&](int3 kij) {
            return ((kij[1]+kij[2]*nb) % n_threads); // IMPORTANT. Every (i,j) should map to a given fixed thread
        })
        .set_binding([&](int3 kij) {
            return true; // IMPORTANT
        })
        .set_name([&](int3 kij) { // This is just for debugging and profiling
            return "ACCU_" + to_string(kij[0]) + "_" + to_string(kij[1])+"_"+to_string(kij[2])+"_"+to_string(comm_rank());
        });

    printf("Starting Cholesky factorization...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    timer t0 = wctime();
    if (rank == 0){
        potrf.fulfill_promise(0);
    }
    tp.join();
    MPI_Barrier(MPI_COMM_WORLD);
    timer t1 = wctime();
    printf("Doen with Cholesky factorization...\n");
    printf("Elapsed time: %e\n", elapsed(t0, t1));
    printf("Potrf time: %e\n", potrf_us_t.load() * 1e-6);
    printf("Trsm time: %e\n", trsm_us_t.load() * 1e-6);
    printf("Gemm time: %e\n", gemm_us_t.load() * 1e-6);
    printf("Accu time: %e\n", accu_us_t.load() * 1e-6);

    std::ofstream logfile;
    string filename = "ttor_distributed_samecol_"+to_string(n)+"_"+to_string(nb)+"_"+ to_string(n_ranks)+".log."+to_string(rank);
    logfile.open(filename);
    logfile << log;
    logfile.close();

    printf("Starting sending matrix to rank 0...\n");
    // Send the matrix to rank 0
    for (int ii=0; ii<nb; ii++) {
        for (int jj=0; jj<nb; jj++) {
            if (jj<=ii)  {
                int owner = block_2_rank(ii,jj);
                MPI_Status status;
                if (rank == 0 && rank != owner) { // Careful with deadlocks here
                    MPI_Recv(blocks[ii+jj*nb]->data(), n*n, MPI_DOUBLE, owner, 0, MPI_COMM_WORLD, &status);
                } else if (rank != 0 && rank == owner) {
                    MPI_Send(blocks[ii+jj*nb]->data(), n*n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                }
            }
        }
    }

    // Rank 0 test
    if(rank == 0) {
        printf("Starting test on rank 0...\n");
        for (int ii=0; ii<nb; ii++) {
            for (int jj=0; jj<nb; jj++) {
                L.block(ii*n,jj*n,n,n)=*blocks[ii+jj*nb];
            }
        }
        auto L1=L.triangularView<Lower>();
        VectorXd x = VectorXd::Random(n * nb);
        VectorXd b = A*x;
        VectorXd bref = b;
        L1.solveInPlace(b);
        L1.transpose().solveInPlace(b);
        double error = (b - x).norm() / x.norm();
        cout << "Error solve: " << error << endl;
        if(error > 1e-10) {
            printf("\n\nERROR: error is too large!\n\n");
        }
    }
}

int main(int argc, char **argv)
{
    int req = MPI_THREAD_FUNNELED;
    int prov = -1;

    MPI_Init_thread(NULL, NULL, req, &prov);

    assert(prov == req);

    int n_threads = 2;
    int verb = 0; // Can be changed to vary the verbosity of the messages
    int n = 1;
    int nb = 2;

    if (argc >= 2)
    {
        n = atoi(argv[1]);
    }

    if (argc >= 3)
    {
        nb = atoi(argv[2]);
    }
    
    if (argc >= 4) {
        n_threads = atoi(argv[3]);
    }
    
    if (argc >= 5) {
        verb = atoi(argv[4]);
    }

    cholesky(n_threads, verb, n, nb);

    MPI_Finalize();
}