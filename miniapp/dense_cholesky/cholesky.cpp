#include "tasktorrent/tasktorrent.hpp"
#ifdef USE_MKL
#include <mkl_cblas.h>
#include <mkl_lapacke.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif
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

typedef array<int, 2> int2;
typedef array<int, 3> int3;


/*
Parametrized priorities for cholesky:
0. No priority, only enforces potrf>trsm>gemm
1. Row-based priority, prioritize tasks with smaller row number in addition to priority 0.
2. Critical path priority, prioritize tasks with longest distance to the exit task. For references, check out the paper
    Beaumont, Olivier, et al. "A Makespan Lower Bound for the Scheduling of the Tiled Cholesky Factorization based on ALAP Schedule." (2020).
3. Critical path and row priority, prioritize tasks with smaller row number in addition to priority 2. We also enforces potrf>trsm>gemm
*/

enum PrioKind { no = 0, row = 1, cp = 2, cp_row = 3};

void cholesky(const int n_threads, const int verb, const int block_size, const int num_blocks, const int nprows, const int npcols, 
              const PrioKind prio_kind, const bool log, const bool deps_log, const bool test, const int accumulate_parallel)
{
    const int rank = comm_rank();
    const int n_ranks = comm_size();
    assert(nprows * npcols == n_ranks);
    std::atomic<long long int> potrf_us_t(0);
    std::atomic<long long int> trsm_us_t(0);
    std::atomic<long long int> gemm_us_t(0);
    std::atomic<long long int> accu_us_t(0);
    
    // Map tasks to ranks
    auto block_2_rank = [&](int i, int j) {
        int r = (j % npcols) * nprows + (i % nprows);
        assert(r >= 0 && r < n_ranks);
        return r;
    };

    // Map threads to ranks
    auto block_2_thread = [&](int i, int j) {
        int ii = i / nprows;
        int jj = j / npcols;
        int num_blocksit = num_blocks / nprows;
        return (ii + jj * num_blocksit) % n_threads;
    };

    // Initializes the matrix
    auto val = [&](int i, int j) { return 1/(float)((i-j)*(i-j)+1); };
    vector<unique_ptr<MatrixXd>> blocks(num_blocks*num_blocks);
    for (int ii=0; ii<num_blocks; ii++) {
        for (int jj=0; jj<num_blocks; jj++) {
            auto val_loc = [&](int i, int j) { return val(ii*block_size+i,jj*block_size+j); };
            if(block_2_rank(ii,jj) == rank) {
                blocks[ii+jj*num_blocks]=make_unique<MatrixXd>(block_size,block_size);
                *blocks[ii+jj*num_blocks]=MatrixXd::NullaryExpr(block_size, block_size, val_loc);
            } else {
                blocks[ii+jj*num_blocks]=make_unique<MatrixXd>();
            }
        }
    }

    // Holds the temporary matrices result of gemm to be accumulated by accu
    // Each block holds data to be accumulated into a given block[ii+jj*num_blocks]
    struct acc_data {
        std::map<int, std::unique_ptr<MatrixXd>> to_accumulate; // to_accumulate[k] holds matrix result of gemm(k,i,j)
        std::mutex mtx; // Protects that map
    };
    std::vector<acc_data> gemm_results(num_blocks*num_blocks); // gemm_results[ii+jj*num_blocks] holds the data to be accumulated into blocks[ii+jj*num_blocks]

    // Set priorities
    auto potf_block_2_prio = [&](int j) {
        if (prio_kind == PrioKind::cp_row) {
            return (double)(9*(num_blocks - j)-1) + 18 * num_blocks * num_blocks;
        }
        else if(prio_kind == PrioKind::cp) {
            return (double)(9*(num_blocks - j)-1);
        } 
        else if(prio_kind == PrioKind::row) {
            return 3.0*(double)(num_blocks-j);
        } 
        else {
            return 3.0;
        }
    };
    auto trsm_block_2_prio = [&](int2 ij) {
        if (prio_kind == PrioKind::cp_row) {
            return (double)((num_blocks - ij[0]) + num_blocks * (9.0 * num_blocks - 9.0 * ij[1] - 2.0) + 9 * num_blocks * num_blocks);
        }
        else if(prio_kind == PrioKind::cp) {
            return (double)(9*(num_blocks - ij[1])-2);
        } 
        else if(prio_kind == PrioKind::row) {
            return 2.0*(double)(num_blocks - ij[0]);
        } 
        else {
            return 2.0;
        }
    };
    auto gemm_block_2_prio = [&](int3 kij) {
        if (prio_kind == PrioKind::cp_row) {
            if (accumulate_parallel) {
                return (double)(num_blocks - kij[1]) + num_blocks * (9.0 * num_blocks - 9.0 * kij[2] - 2.0);
            }
            else {
                return (double)(num_blocks - kij[1]) + num_blocks * (9.0 * num_blocks - 3.0 * kij[2] - 6.0 * kij[0] - 2.0);
            }
        }
        else if(prio_kind == PrioKind::cp) {
            return (double)(9*num_blocks-9*kij[2]-2);
        } 
        else if(prio_kind == PrioKind::row) {
            return (double)(num_blocks - kij[1]);
        } 
        else {
            return 1.0;
        }
    };
    // Names
    auto potrf_name = [](int j, int r) {
        return "POTRF_" + to_string(j) + "_r" + to_string(r);
    };
    auto trsm_name = [](int2 ij, int r) {
        return "TRSM_" + to_string(ij[0]) + "_" + to_string(ij[1]) + "_r" + to_string(r);
    };
    auto gemm_name = [](int3 kij, int r) {
        return "GEMM_" + to_string(kij[0]) + "_" + to_string(kij[1]) + "_" + to_string(kij[2]) + "_r" + to_string(r);
    };
    auto accu_name = [](int3 kij, int r) {
        return "ACCU_" + to_string(kij[0]) + "_" + to_string(kij[1]) + "_" + to_string(kij[2]) + "_r" + to_string(r);
    };

    const int num_blocksmax = 15;
    MPI_Barrier(MPI_COMM_WORLD);
    if(comm_rank() == 0) {
        printf("Block -> Rank\n");
        for(int i = 0; i < min(num_blocksmax, num_blocks); i++) {
            for(int j = 0; j < min(num_blocksmax, num_blocks); j++) {
                if(i >= j) {
                    printf("%2d ", block_2_rank(i, j));
                }
            }
            printf("\n");
        }
        printf("Potf/trsm -> Priority\n");
        for(int i = 0; i < min(num_blocksmax, num_blocks); i++) {
            for(int j = 0; j < min(num_blocksmax, num_blocks); j++) {
                if(i == j) {
                    printf("%5f ", potf_block_2_prio(i));
                } else if (i > j) {
                    printf("%5f ", trsm_block_2_prio({i,j}));
                };
            }
            printf("\n");
        }
        printf("Gemm -> Priority\n");
        for(int k = 0; k < min(num_blocksmax, num_blocks); k++) {
            printf("k = %d\n", k);
            for(int i = 0; i < min(num_blocksmax, num_blocks); i++) {
                for(int j = 0; j < min(num_blocksmax, num_blocks); j++) {
                    if(i >= j) {
                        if(k < j) {
                            printf("%5f ", gemm_block_2_prio({k,i,j}));
                        } else {
                            printf(".     ");
                        }
                    }
                }
                printf("\n");
            }
        }
    }
    for(int r = 0; r < ttor::comm_size(); r++) {
        if(r == comm_rank()) {
            printf("[%d] Block -> thread\n", r);
            for(int i = 0; i < min(num_blocksmax, num_blocks); i++) {
                for(int j = 0; j < min(num_blocksmax, num_blocks); j++) {
                    if(i >= j && block_2_rank(i,j) == r) {
                        printf("%2d ", block_2_thread(i, j));
                    } else {
                        printf(" . ");
                    }
                }
                printf("\n");
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Initialize the communicator structure
    Communicator comm(verb);

    // Initialize the runtime structures
    Threadpool tp(n_threads, &comm, verb, "Wk_Chol_" + to_string(rank) + "_");
    Taskflow<int>  potrf(&tp, verb);
    Taskflow<int2> trsm(&tp, verb);
    Taskflow<int3> gemm(&tp, verb);
    Taskflow<int3> accu(&tp, verb);

    Logger logger(1000000);
    if(log) {
        tp.set_logger(&logger);
        comm.set_logger(&logger);
    }

    DepsLogger dlog(1000000);

    // Send a potrf'ed pivot A(k,k) and trigger trsms below requiring A(k,k)
    auto am_trsm = comm.make_active_msg( 
            [&](view<double> &Ljj, int& j, view<int>& is) {
                *blocks[j+j*num_blocks] = Map<MatrixXd>(Ljj.data(), block_size, block_size);
                for(auto& i: is) {
                    trsm.fulfill_promise({i,j});
                }
            });

    /**
     * j is the pivot's position at A(j,j)
     */
    potrf.set_task([&](int j) { // A[j,j] -> A[j,j]
            assert(block_2_rank(j,j) == rank);
            timer t_ = wctime();
            LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', block_size, blocks[j+j*num_blocks]->data(), block_size);
            timer t__ = wctime();
            potrf_us_t += 1e6 * elapsed(t_, t__);
        })
        .set_fulfill([&](int j) { // Triggers all trsms on rows i > j, A[i,j]
            assert(block_2_rank(j,j) == rank);
            map<int,vector<int>> fulfill;
            for (int i = j+1; i<num_blocks; i++) {
                fulfill[block_2_rank(i,j)].push_back(i);
            }
            for (auto& rf: fulfill) {
                int r = rf.first;
                if (rank == r) {
                    for (auto& i: rf.second) {
                        trsm.fulfill_promise({i,j});
                        if(deps_log) {
                            dlog.add_event(make_unique<DepsEvent>(potrf.name(j), trsm.name({i,j})));
                        }
                    }
                } else {
                    auto Ljjv = view<double>(blocks[j+j*num_blocks]->data(), block_size*block_size);
                    auto isv = view<int>(rf.second.data(), rf.second.size());                    
                    if(deps_log) {
                        for(auto i: isv) {
                            dlog.add_event(make_unique<DepsEvent>(potrf.name(j), trsm_name({i,j}, r)));
                        }
                    }
                    am_trsm->send(r, Ljjv, j, isv);
                }

            }
        })
        .set_indegree([&](int j) {
            assert(block_2_rank(j,j) == rank);
            if(accumulate_parallel) {
                return j == 0 ? 1 : j; // Need j accumulations into (j,j) to trigger the potf
            } else {
                return 1;
            }
        })
        .set_priority(potf_block_2_prio)
        .set_mapping([&](int j) {
            assert(block_2_rank(j,j) == rank);
            return block_2_thread(j, j);
        })
        .set_name([&](int j) { // This is just for debugging and profiling
            assert(block_2_rank(j,j) == rank);
            return potrf_name(j, rank);
        });

    // Sends a panel (trsm'ed block A(i,j)) and trigger gemms requiring A(i,j)
    auto am_gemm = comm.make_active_msg(
        [&](view<double> &Lij, int& i, int& j, view<int2>& ijs) {
            *blocks[i+j*num_blocks] = Map<MatrixXd>(Lij.data(), block_size, block_size);
            for(auto& ij: ijs) {
                gemm.fulfill_promise({j,ij[0],ij[1]});
            }
        });

    /**
     * ij is (Row, Col) of the block in the matrix at A(i,j)
     **/
    trsm.set_task([&](int2 ij) { // A[j,j] & A[i,j] -> A[i,j]
            int i=ij[0]; 
            int j=ij[1]; 
            assert(block_2_rank(i,j) == rank);
            assert(i > j);
            timer t_ = wctime();
            cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit, block_size, block_size, 1.0, blocks[j+j*num_blocks]->data(), block_size, blocks[i+j*num_blocks]->data(), block_size);
            timer t__ = wctime();
            trsm_us_t += 1e6 * elapsed(t_, t__);
        })
        .set_fulfill([&](int2 ij) {
            int i=ij[0];
            int j=ij[1];
            assert(block_2_rank(i,j) == rank);
            assert(i > j);
            map<int,vector<int2>> fulfill;
            for (int k = j+1; k < num_blocks; k++) {
                int ii = std::max(i,k);
                int jj = std::min(i,k);
                fulfill[block_2_rank(ii,jj)].push_back({ii,jj});
            }
            for (auto& rf: fulfill) {
                int r = rf.first;
                if (r == rank) {
                    for (auto& ij_gemm : rf.second) {
                        gemm.fulfill_promise({j,ij_gemm[0],ij_gemm[1]});
                        if(deps_log) {
                            dlog.add_event(make_unique<DepsEvent>(trsm.name(ij), gemm.name({j,ij_gemm[0],ij_gemm[1]})));
                        }
                    }
                }
                else {
                    auto Lijv = view<double>(blocks[i+j*num_blocks]->data(), block_size*block_size);
                    auto ijsv = view<int2>(rf.second.data(), rf.second.size());
                    for(auto ij_gemm: ijsv) {
                        if(deps_log) {
                            if(deps_log) {
                                dlog.add_event(make_unique<DepsEvent>(trsm.name(ij), gemm_name({j,ij_gemm[0],ij_gemm[1]}, r)));
                            }
                        }
                    }
                    am_gemm->send(r, Lijv, i, j, ijsv);
                }
            }
        })
        .set_indegree([&](int2 ij) {
            assert(block_2_rank(ij[0],ij[1]) == rank);
            if(accumulate_parallel) {
                return 1 + ij[1]; // Potrf above and all gemms before
            } else {
                return 1 + (ij[1] == 0 ? 0 : 1); // Potrf and last gemm before
            }
        })
        .set_priority(trsm_block_2_prio)
        .set_mapping([&](int2 ij) {
            assert(block_2_rank(ij[0],ij[1]) == rank);
            return block_2_thread(ij[0], ij[1]);
        })
        .set_name([&](int2 ij) { // This is just for debugging and profiling
            assert(block_2_rank(ij[0],ij[1]) == rank);
            return trsm_name(ij, rank);
        });

    /**
     * k is the step (the pivot's position), ij are Row and Column, at A(i,j)
     **/
    gemm.set_task([&](int3 kij) {
            assert(block_2_rank(kij[1],kij[2]) == rank);
            const int k=kij[0];
            const int i=kij[1];
            const int j=kij[2];
            assert(j <= i);
            assert(k < j);
            std::unique_ptr<MatrixXd> Atmp;
            MatrixXd* Aij;
            double beta = 1.0;
            if(accumulate_parallel) {
                beta = 0.0;
                Atmp = make_unique<MatrixXd>(block_size, block_size); // The matrix is allocated with garbage. The 0 in the BLAS call make sure its overwritten by 0's before doing any math
                Aij = Atmp.get();
            } else {
                beta = 1.0;
                Aij = blocks[i+j*num_blocks].get();
            }
            timer t_ = wctime();
            if (i == j) {
                cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans, block_size, block_size, -1.0, blocks[i+k*num_blocks]->data(), block_size, beta, Aij->data(), block_size);
            } else {
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, block_size, block_size, block_size, -1.0,blocks[i+k*num_blocks]->data(), block_size, blocks[j+k*num_blocks]->data(), block_size, beta, Aij->data(), block_size);
            }
            timer t__ = wctime();
            gemm_us_t += 1e6 * elapsed(t_, t__);
            if(accumulate_parallel) {
                lock_guard<mutex> lock(gemm_results[i+j*num_blocks].mtx);
                gemm_results[i+j*num_blocks].to_accumulate[k] = move(Atmp);
            }
        })
        .set_fulfill([&](int3 kij) {
            const int k=kij[0];
            const int i=kij[1];
            const int j=kij[2];
            assert(block_2_rank(kij[1],kij[2]) == rank);
            if(accumulate_parallel) {
                if(deps_log) {
                    dlog.add_event(make_unique<DepsEvent>(gemm.name(kij), accu.name(kij)));
                }
                accu.fulfill_promise(kij);
            } else {
                if (k < j-1) {
                    gemm.fulfill_promise({k+1, i, j});
                    if(deps_log) {
                        dlog.add_event(make_unique<DepsEvent>(gemm.name(kij), gemm.name({k+1, i, j})));
                    }
                } else {
                    if (i == j) {
                        if(deps_log) {
                            dlog.add_event(make_unique<DepsEvent>(gemm.name(kij), potrf.name(i)));
                        }
                        potrf.fulfill_promise(i);
                    } else {
                        if(deps_log) {
                            dlog.add_event(make_unique<DepsEvent>(gemm.name(kij), trsm.name({i,j})));
                        }
                        trsm.fulfill_promise({i,j});
                    }
                }
            }
        })
        .set_indegree([&](int3 kij) {
            assert(block_2_rank(kij[1],kij[2]) == rank);
            if(accumulate_parallel) {
                return kij[1] == kij[2] ? 1 : 2; // Either one potf or two trsms
            } else {
                return (kij[1] == kij[2] ? 1 : 2) + (kij[0] == 0 ? 0 : 1); // one potrf or two trsms + the gemm before
            }
        })
        .set_priority(gemm_block_2_prio)
        .set_mapping([&](int3 kij) {
            assert(block_2_rank(kij[1],kij[2]) == rank);
            return block_2_thread(kij[1], kij[2]); // IMPORTANT if accumulate_parallel is true
        })
        .set_binding([&](int3 kij) {
            return false; // If we accumulate in parallel, there is no order for the gemm so it doesnt matter ; If we don't then we do the gemm in sequence anyway
        }).set_name([&](int3 kij) { // This is just for debugging and profiling
            assert(block_2_rank(kij[1],kij[2]) == rank);
            return gemm_name(kij, rank);
        });

    /**
     * k is the step (the pivot's position), ij are Row and Column, at A(i,j)
     **/
    accu.set_task([&](int3 kij) {
            assert(block_2_rank(kij[1],kij[2]) == rank);
            int k=kij[0]; // Step (gemm's pivot)
            int i=kij[1]; // Row
            int j=kij[2]; // Col
            assert(j <= i);
            assert(k < j);
            std::unique_ptr<Eigen::MatrixXd> Atmp;
            {
                lock_guard<mutex> lock(gemm_results[i+j*num_blocks].mtx);
                Atmp = move(gemm_results[i+j*num_blocks].to_accumulate[k]);
                gemm_results[i+j*num_blocks].to_accumulate.erase(k);
            }
            timer t_ = wctime();
            *blocks[i+j*num_blocks] += (*Atmp);
            timer t__ = wctime();
            accu_us_t += 1e6 * elapsed(t_, t__);
        })
        .set_fulfill([&](int3 kij) {
            assert(block_2_rank(kij[1],kij[2]) == rank);
            int k=kij[0];
            int i=kij[1];
            int j=kij[2];
            assert(j <= i);
            assert(k < j);
            if(i == j) {
                if(deps_log) {
                    dlog.add_event(make_unique<DepsEvent>(accu.name(kij), potrf.name(i)));
                }
                potrf.fulfill_promise(i);
            } else {
                trsm.fulfill_promise({i,j});
                if(deps_log) {
                    dlog.add_event(make_unique<DepsEvent>(accu.name(kij), trsm.name({i,j})));
                }
            }
        })
        .set_indegree([&](int3 kij) {
            assert(block_2_rank(kij[1],kij[2]) == rank);
            return 1;
        })
        .set_mapping([&](int3 kij) {
            assert(block_2_rank(kij[1],kij[2]) == rank);
            return block_2_thread(kij[1], kij[2]); // IMPORTANT. Every (i,j) should map to a given fixed thread
        })
        .set_priority(gemm_block_2_prio)
        .set_binding([&](int3 kij) {
            assert(block_2_rank(kij[1],kij[2]) == rank);
            return true; // IMPORTANT
        })
        .set_name([&](int3 kij) { // This is just for debugging and profiling
            assert(block_2_rank(kij[1],kij[2]) == rank);
            return accu_name(kij, rank);
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
    double total_time = elapsed(t0, t1);
    printf("Done with Cholesky factorization...\n");
    printf("Elapsed time: %e\n", total_time);
    printf("Potrf time: %e\n", potrf_us_t.load() * 1e-6);
    printf("Trsm time: %e\n", trsm_us_t.load() * 1e-6);
    printf("Gemm time: %e\n", gemm_us_t.load() * 1e-6);
    printf("Accu time: %e\n", accu_us_t.load() * 1e-6);

    printf("++++rank,nranks,n_threads,matrix_size,block_size,num_blocks,priority_kind,accumulate,total_time\n");
    printf("[%d]>>>>%d,%d,%d,%d,%d,%d,%d,%d,%e\n",rank,rank,n_ranks,n_threads,block_size*num_blocks,block_size,num_blocks,(int)prio_kind,(int)accumulate_parallel,total_time);

    if(log) {
        std::ofstream logfile;
        string filename = "ttor_dist_"+to_string(block_size)+"_"+to_string(num_blocks)+"_"+ to_string(n_ranks)+"_"+to_string(n_threads)+"_"+to_string(prio_kind)+".log."+to_string(rank);
        logfile.open(filename);
        logfile << logger;
        logfile.close();
    }

    if(deps_log) {
        std::ofstream depsfile;
        string depsfilename = "deps_ttor_dist_"+to_string(block_size)+"_"+to_string(num_blocks)+"_"+ to_string(n_ranks)+"_"+to_string(n_threads)+"_"+to_string(prio_kind)+".dot."+to_string(rank);
        depsfile.open(depsfilename);
        depsfile << dlog;
        depsfile.close();
    }

    if(test) {
        printf("Starting sending matrix to rank 0...\n");
    	MatrixXd A;
    	A = MatrixXd::NullaryExpr(block_size*num_blocks,block_size*num_blocks,val);
    	MatrixXd L = A;
        // Send the matrix to rank 0
        for (int ii=0; ii<num_blocks; ii++) {
            for (int jj=0; jj<num_blocks; jj++) {
                if (jj<=ii)  {
                    int owner = block_2_rank(ii,jj);
                    MPI_Status status;
                    if (rank == 0 && rank != owner) { // Careful with deadlocks here
                        blocks[ii+jj*num_blocks] = make_unique<Eigen::MatrixXd>(block_size, block_size);
                        MPI_Recv(blocks[ii+jj*num_blocks]->data(), block_size*block_size, MPI_DOUBLE, owner, 0, MPI_COMM_WORLD, &status);
                    } else if (rank != 0 && rank == owner) {
                        MPI_Send(blocks[ii+jj*num_blocks]->data(), block_size*block_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                    }
                }
            }
        }

        // Rank 0 test
        if(rank == 0) {
            printf("Starting test on rank 0...\n");
            for (int ii=0; ii<num_blocks; ii++) {
                for (int jj=0; jj<num_blocks; jj++) {
                    if (jj<=ii) {
                        L.block(ii*block_size,jj*block_size,block_size,block_size)=*blocks[ii+jj*num_blocks];
                    }
                }
            }
            auto L1=L.triangularView<Lower>();
            VectorXd x = VectorXd::Random(block_size * num_blocks);
            VectorXd b = A*x;
            VectorXd bref = b;
            L1.solveInPlace(b);
            L1.transpose().solveInPlace(b);
            double error = (b - x).norm() / x.norm();
            cout << "Error solve: " << error << endl;
            if(error > 1e-10) {
                printf("\n\nERROR: error is too large!\n\n");
                exit(1);
            }
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
    int block_size = 5;
    int num_blocks = 10;
    int nprows = 1;
    int npcols = ttor::comm_size();
    PrioKind kind = PrioKind::no;
    bool log = false;
    bool depslog = false;
    bool test = true;
    bool accumulate = false;

    if (argc >= 2)
    {
        block_size = atoi(argv[1]);
        assert(block_size > 0);
    }

    if (argc >= 3)
    {
        num_blocks = atoi(argv[2]);
        assert(num_blocks > 0);
    }
    
    if (argc >= 4) {
        n_threads = atoi(argv[3]);
        assert(n_threads > 0);
    }
    
    if (argc >= 5) {
        verb = atoi(argv[4]);
        assert(verb >= 0);
    }

    if (argc >= 6) {
        nprows = atoi(argv[5]);
        assert(nprows >= 0);
    }

    if (argc >= 7) {
        npcols = atoi(argv[6]);
        assert(npcols >= 0);
    }

    if (argc >= 8) {
        assert(atoi(argv[7]) >= 0 && atoi(argv[7]) < 4);
        kind = (PrioKind)atoi(argv[7]);
    }

    if (argc >= 9) {
        log = static_cast<bool>(atoi(argv[8]));
    }

    if (argc >= 10) {
        depslog = static_cast<bool>(atoi(argv[9]));
    }

    if (argc >= 11) {
        test = static_cast<bool>(atoi(argv[10]));
    }

    if (argc >= 12) {
        accumulate = static_cast<bool>(atoi(argv[11]));
    }

    printf("Usage: ./cholesky block_size num_blocks n_threads verb nprows npcols kind log depslog test accumulate\n");
    printf("Arguments: block_size (size of blocks) %d\nnum_blocks (# of blocks) %d\nn_threads %d\nverb %d\nnprows %d\nnpcols %d\nkind %d\nlog %d\ndeplog %d\ntest %d\naccumulate %d\n", block_size, num_blocks, n_threads, verb, nprows, npcols, (int)kind, log, depslog, test, accumulate);

    cholesky(n_threads, verb, block_size, num_blocks, nprows, npcols, kind, log, depslog, test, accumulate);

    MPI_Finalize();
}
