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
#include <set>

#include <mpi.h>
#include <cxxopts.hpp>

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
              const PrioKind prio_kind, const bool log, const bool deps_log, const bool test, const int accumulate_parallel, const int upper_block_size)
{
    const int rank = comm_rank();
    const int n_ranks = comm_size();
    const int matrix_size = block_size * num_blocks;
    assert(nprows * npcols == n_ranks);
    std::atomic<long long int> potrf_us_t(0);
    std::atomic<long long int> trsm_us_t(0);
    std::atomic<long long int> gemm_us_t(0);
    std::atomic<long long int> accu_us_t(0);

    // Warmup MKL
    {
        Eigen::MatrixXd A = Eigen::MatrixXd::Identity(256,256);
        Eigen::MatrixXd B = Eigen::MatrixXd::Identity(256,256);
        Eigen::MatrixXd C = Eigen::MatrixXd::Identity(256,256);
        for(int i = 0; i < 10; i++) {
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 256, 256, 256, 1.0, A.data(), 256, B.data(), 256, 1.0, C.data(), 256);
        }
    }

    // Compute random sizes
    std::mt19937 gen(2020);
    assert(upper_block_size <= 2*block_size);
    const int lower_block_size = 2*block_size - upper_block_size;
    std::uniform_int_distribution<> distrib(lower_block_size,upper_block_size); // average is block_size
    if(rank == 0) printf("lower_block_size %d, upper_block_size %d\n", lower_block_size, upper_block_size);
    std::vector<int> block_sizes(num_blocks, block_size);
    {
        int n = 0;
        for(int i = 0; i < num_blocks-1; i++) {
            int bs = std::min(matrix_size - n, distrib(gen));
            n += bs;
            block_sizes[i] = bs;
        }
        assert(matrix_size - n >= 0);
        block_sizes[num_blocks-1] = matrix_size - n;
    }
    int total = std::accumulate(block_sizes.begin(), block_sizes.end(), 0);
    assert(total == matrix_size);
    std::vector<int> block_displ(num_blocks+1, 0);
    for(int i = 1; i < num_blocks+1; i++) {
        block_displ[i] = block_displ[i-1] + block_sizes[i-1];
    }
    std::vector<int> block_sizes_lda(num_blocks, 0);
    for(int i = 0; i < num_blocks; i++) {
        block_sizes_lda[i] = std::max(1, block_sizes[i]);
    }
    assert(block_displ[num_blocks] == matrix_size);
    if(rank == 0) {
        printf("block sizes: ");
        for(int i = 0; i < num_blocks; i++) { 
            assert(block_sizes[i] >= 0);
            printf("%d ", block_sizes[i]); 
        };
        printf("\n");
        printf("block displ: ");
        for(int i = 0; i < num_blocks+1; i++) { 
            assert(block_displ[i] >= 0);
            printf("%d ", block_displ[i]); 
        };
        printf("\n");
    }
    
    // Map tasks to ranks
    auto block_2_rank = [&](int i, int j) {
        assert(i >= 0 && i < num_blocks);
        assert(j >= 0 && j < num_blocks);
        int r = (j % npcols) * nprows + (i % nprows);
        assert(r >= 0 && r < n_ranks);
        return r;
    };

    const int rank_row = (rank % nprows);
    const int rank_col = (rank / nprows);
    auto block_2_rank_row = [&](int i, int j) {
        return i % nprows;
    };
    auto block_2_rank_col = [&](int i, int j) {
        return j % npcols;
    };

    // Map threads to ranks
    auto block_2_thread = [&](int i, int j) {
        int ii = i / nprows;
        int jj = j / npcols;
        int num_blocksit = num_blocks / nprows;
        return (ii + jj * num_blocksit) % n_threads;
    };

    // Initializes the matrix
    auto val = [&](int i, int j) { return 1/(double)((i-j)*(i-j)+1); };
    vector<unique_ptr<MatrixXd>> blocks(num_blocks*num_blocks);
    for (int ii=0; ii<num_blocks; ii++) {
        for (int jj=0; jj<num_blocks; jj++) {
            auto val_loc = [&](int i, int j) { return val(block_displ[ii]+i,block_displ[jj]+j); };
            if(ii >= jj) {
                if(block_2_rank(ii,jj) == rank) {
                    blocks[ii+jj*num_blocks]=make_unique<MatrixXd>(block_sizes[ii],block_sizes[jj]);
                    *blocks[ii+jj*num_blocks]=MatrixXd::NullaryExpr(block_sizes[ii],block_sizes[jj], val_loc);
                } else {
                    blocks[ii+jj*num_blocks]=make_unique<MatrixXd>(0,0);
                }
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
    auto potrf_name = [](int j) {
        return "POTRF_" + to_string(j);
    };
    auto trsm_name = [](int2 ij) {
        return "TRSM_" + to_string(ij[0]) + "_" + to_string(ij[1]);
    };
    auto gemm_name = [](int3 kij) {
        return "GEMM_" + to_string(kij[0]) + "_" + to_string(kij[1]) + "_" + to_string(kij[2]);
    };
    auto accu_name = [](int3 kij) {
        return "ACCU_" + to_string(kij[0]) + "_" + to_string(kij[1]) + "_" + to_string(kij[2]);
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
    Communicator comm(MPI_COMM_WORLD, verb);

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
    auto am_trsm = comm.make_large_active_msg( 
            [&](int& j) {
                int off = (nprows + rank_row - block_2_rank_row(j,j)) % nprows;
                assert(off > 0); // Can't be me
                assert(off < nprows);
                for (int i = j + off; i < num_blocks; i += nprows) {
                    assert(block_2_rank(i,j) == rank);
                    trsm.fulfill_promise({i,j});
                }
            },
            [&](int& j) {
                blocks[j+j*num_blocks]->resize(block_sizes[j],block_sizes[j]);
                return blocks[j+j*num_blocks]->data();
            },
            [&](int&){
                return;
            });

    /**
     * j is the pivot's position at A(j,j)
     */
    potrf.set_task([&](int j) { // A[j,j] -> A[j,j]
            assert(block_2_rank(j,j) == rank);
            timer t_ = wctime();
            LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', block_sizes[j], blocks[j+j*num_blocks]->data(), block_sizes_lda[j]);
            timer t__ = wctime();
            potrf_us_t += 1e6 * elapsed(t_, t__);
        })
        .set_fulfill([&](int j) { // Triggers all trsms on rows i > j, A[i,j]
            assert(block_2_rank(j,j) == rank);
            if(deps_log) {
                for(int i = j+1; i < num_blocks; i++) {
                    dlog.add_event(make_unique<DepsEvent>(potrf.name(j), trsm.name({i,j})));
                }
            }
            // Trigger myself
            for (int i = j + nprows; i < num_blocks; i += nprows) {
                assert(block_2_rank(i,j) == rank);
                trsm.fulfill_promise({i,j});
            }
            // Send to other procs in column
            auto Ljjv = view<double>(blocks[j+j*num_blocks]->data(), block_sizes[j]*block_sizes[j]);
            for(int p = 0; p < nprows; p++) {
                if(j+p >= num_blocks) break;
                int dest = block_2_rank(j+p,j);
                if(dest != rank) {
                    am_trsm->send_large(dest, Ljjv, j);
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
            return potrf_name(j);
        });

    // Sends a panel (trsm'ed block A(i,j)) and trigger gemms requiring A(i,j)
    auto am_gemm = comm.make_large_active_msg(
        [&](int& i, int& j) {
            if(block_2_rank_row(i,j) == rank_row) {
                const int off_right = (npcols + rank_col - block_2_rank_col(i,j)) % npcols;
                assert(off_right > 0); // Can't be me
                assert(off_right < npcols);
                assert(i >= 0 && i < num_blocks);
                assert(j >= 0 && j < num_blocks);
                for (int k = j + off_right; k < i; k += npcols) {
                    assert(block_2_rank(i,k) == rank);
                    gemm.fulfill_promise({j,i,k});
                }
            }
            if(block_2_rank_col(i,i) == rank_col) {
                const int off_below = (nprows + rank_row - block_2_rank_row(i,i)) % nprows;
                assert(off_below >= 0); // Could be me
                assert(off_below < nprows);
                for (int k = i + off_below; k < num_blocks; k += nprows) {
                    assert(block_2_rank(k,i) == rank);
                    gemm.fulfill_promise({j,k,i});
                }
            }
        },
        [&](int& i, int& j) {
            blocks[i+j*num_blocks]->resize(block_sizes[i],block_sizes[j]);
            return blocks[i+j*num_blocks]->data();
        },
        [&](int& i, int& j) {
            return;
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
            cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit, 
                block_sizes[i], block_sizes[j], 1.0, blocks[j+j*num_blocks]->data(), block_sizes_lda[j], blocks[i+j*num_blocks]->data(), block_sizes_lda[i]);
            timer t__ = wctime();
            trsm_us_t += 1e6 * elapsed(t_, t__);
        })
        .set_fulfill([&](int2 ij) {
            int i=ij[0];
            int j=ij[1];
            assert(block_2_rank(i,j) == rank);
            assert(i > j);
            if(deps_log) {
                for(int k = j+1; k < num_blocks; k++) {
                    dlog.add_event(make_unique<DepsEvent>(trsm.name(ij), gemm.name({j,std::max(i,k),std::min(i,k)})));
                }
            }
            // Local
            // Careful to not count the pivot (syrk) twice
            for (int k = j + npcols; k < i; k += npcols) {
                assert(block_2_rank(i,k) == rank);
                gemm.fulfill_promise({j,i,k});
            }
            if(block_2_rank_col(i,i) == rank_col) {
                int off_below = (nprows + rank_row - block_2_rank_row(i,i)) % nprows;
                for (int k = i + off_below; k < num_blocks; k += nprows) {
                    assert(block_2_rank(k,i) == rank);
                    gemm.fulfill_promise({j,k,i});
                }
            }
            // Remote
            auto Lijv = view<double>(blocks[i+j*num_blocks]->data(), block_sizes[i]*block_sizes[j]);
            std::set<int> dests;
            for (int c = 0; c < npcols; c++) {
                if(j+c >= num_blocks) break;
                int dest = block_2_rank(i,j+c);
                if(dest != rank) dests.insert(dest);
            }
            for (int r = 0; r < nprows; r++) {
                if(i+r >= num_blocks) break;
                int dest = block_2_rank(i+r,i);
                if(dest != rank) dests.insert(dest);
            }
            for(auto& dest: dests) {
                am_gemm->send_large(dest, Lijv, i, j);
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
            return trsm_name(ij);
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
                Atmp = make_unique<MatrixXd>(block_sizes[i], block_sizes[j]); // The matrix is allocated with garbage. The 0 in the BLAS call make sure its overwritten by 0's before doing any math
                Aij = Atmp.get();
            } else {
                beta = 1.0;
                Aij = blocks[i+j*num_blocks].get();
            }
            assert(Aij->rows() == block_sizes[i] && Aij->cols() == block_sizes[j]);
            timer t_ = wctime();
            if (i == j) {
                cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans, 
                    block_sizes[i], block_sizes[k], -1.0, blocks[i+k*num_blocks]->data(), block_sizes_lda[i], beta, Aij->data(), block_sizes_lda[i]);
            } else {
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, 
                    block_sizes[i], block_sizes[j], block_sizes[k], -1.0, blocks[i+k*num_blocks]->data(), block_sizes_lda[i], blocks[j+k*num_blocks]->data(), block_sizes_lda[j], beta, Aij->data(), block_sizes_lda[i]);
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
                    if(deps_log) {
                        dlog.add_event(make_unique<DepsEvent>(gemm.name(kij), gemm.name({k+1, i, j})));
                    }
                    gemm.fulfill_promise({k+1, i, j});
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
            return gemm_name(kij);
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
                if(deps_log) {
                    dlog.add_event(make_unique<DepsEvent>(accu.name(kij), trsm.name({i,j})));
                }
                trsm.fulfill_promise({i,j});
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
            return accu_name(kij);
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

    printf("++++rank nranks n_threads matrix_size block_size num_blocks priority_kind accumulate upper_block_size total_time\n");
    printf("[%d]>>>>%d %d %d %d %d %d %d %d %d %e\n",rank,rank,n_ranks,n_threads,matrix_size,block_size,num_blocks,(int)prio_kind,(int)accumulate_parallel,upper_block_size,total_time);

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
        MatrixXd A = MatrixXd::NullaryExpr(matrix_size,matrix_size,val);
        MatrixXd L = MatrixXd::Zero(matrix_size,matrix_size);
        // Send the matrix to rank 0
        for (int ii=0; ii<num_blocks; ii++) {
            for (int jj=0; jj<num_blocks; jj++) {
                if (jj<=ii)  {
                    int owner = block_2_rank(ii,jj);
                    MPI_Status status;
                    if (rank == 0 && rank != owner) { // Careful with deadlocks here
                        blocks[ii+jj*num_blocks] = make_unique<Eigen::MatrixXd>(block_sizes[ii], block_sizes[jj]);
                        MPI_Recv(blocks[ii+jj*num_blocks]->data(), block_sizes[ii]*block_sizes[jj], MPI_DOUBLE, owner, 0, MPI_COMM_WORLD, &status);
                    } else if (rank != 0 && rank == owner) {
                        MPI_Send(blocks[ii+jj*num_blocks]->data(), block_sizes[ii]*block_sizes[jj], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
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
                        L.block(block_displ[ii],block_displ[jj],block_sizes[ii],block_sizes[jj])=*blocks[ii+jj*num_blocks];
                    }
                }
            }
            auto L1=L.triangularView<Lower>();
            VectorXd x = VectorXd::Random(matrix_size);
            VectorXd b = A*x;
            VectorXd bref = b;
            L1.solveInPlace(b);
            L1.transpose().solveInPlace(b);
            double error = (b - x).norm() / x.norm();
            printf("\n=> Error solve %e\n\n", error);
            if(error > 1e-6) {
                printf("\n\nERROR: error is too large!\n\n");
                exit(1);
            }
        }
    }
}

int main(int argc, const char **argv)
{
    int req = MPI_THREAD_FUNNELED;
    int prov = -1;

    MPI_Init_thread(NULL, NULL, req, &prov);

    assert(prov == req);

    std::stringstream sstr;
    sstr << comm_size();
    const std::string comm_size_str = sstr.str();

    cxxopts::Options options("2d_cholesky", "2D dense cholesky using TaskTorrent");
    options.add_options()
        ("help", "Print help")
        ("n_threads", "Number of threads", cxxopts::value<int>()->default_value("2"))
        ("verb", "Verbosity level", cxxopts::value<int>()->default_value("0"))
        ("block_size", "Block size", cxxopts::value<int>()->default_value("5"))
        ("num_blocks", "Number of blocks", cxxopts::value<int>()->default_value("10"))
        ("nprows", "Number of processors accross rows", cxxopts::value<int>()->default_value("1"))
        ("npcols", "Number of processors accross columns", cxxopts::value<int>()->default_value(comm_size_str.c_str()))
        ("kind", "Priority kind", cxxopts::value<int>()->default_value("0"))
        ("log", "Enable logging", cxxopts::value<bool>()->default_value("false"))
        ("depslog", "Enable dependency logging", cxxopts::value<bool>()->default_value("false"))
        ("test", "Test or not", cxxopts::value<bool>()->default_value("true"))
        ("accumulate", "Accumulate block GEMMs in parallel", cxxopts::value<bool>()->default_value("false"))
        ("upper_block_size", "Upper block size", cxxopts::value<int>()->default_value("-1"));
    auto result = options.parse(argc, argv);

    const int n_threads = result["n_threads"].as<int>();
    const int verb = result["verb"].as<int>();
    const int block_size = result["block_size"].as<int>();
    const int num_blocks = result["num_blocks"].as<int>();
    const int nprows = result["nprows"].as<int>();
    const int npcols = result["npcols"].as<int>();
    const PrioKind kind = (PrioKind) result["kind"].as<int>();
    const bool log = result["log"].as<bool>();
    const bool depslog = result["depslog"].as<bool>();
    const bool test = result["test"].as<bool>();
    const bool accumulate = result["accumulate"].as<bool>();
    const int upper_block_size = (result["upper_block_size"].as<int>() == -1 ? block_size : result["upper_block_size"].as<int>());

    assert(block_size > 0);
    assert(num_blocks > 0);
    assert(n_threads > 0);
    assert(verb >= 0);
    assert(nprows >= 0);
    assert(npcols >= 0);
    assert(upper_block_size >= block_size && upper_block_size <= 2 * block_size);

    if (result.count("help")) {
        std::cout << options.help({"", "Group"}) << endl;
        exit(0);
    }
    if(comm_rank() == 0) printf("Arguments: block_size (size of blocks) %d\nnum_blocks (# of blocks) %d\nn_threads %d\nverb %d\nnprows %d\nnpcols %d\nkind %d\nlog %d\ndeplog %d\ntest %d\naccumulate %d\nupper_block_size %d\n", block_size, num_blocks, n_threads, verb, nprows, npcols, (int)kind, log, depslog, test, accumulate, upper_block_size);

    cholesky(n_threads, verb, block_size, num_blocks, nprows, npcols, kind, log, depslog, test, accumulate, upper_block_size);

    MPI_Finalize();
}
