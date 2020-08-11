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

using namespace std;
using namespace Eigen;
using namespace ttor;

typedef array<int, 2> int2;
typedef array<int, 3> int3;

/*
 * Parametrized priorities for cholesky:
 * 0. No priority, only enforces potrf>trsm>gemm
 * 1. Row-based priority, prioritize tasks with smaller row number in addition to priority 0.
 * 2. Critical path priority, prioritize tasks with longest distance to the exit task. For references, check out the paper
	 Beaumont, Olivier, et al. "A Makespan Lower Bound for the Scheduling of the Tiled Cholesky Factorization based on ALAP Schedule." (2020).
 * 3. Critical path and row priority, prioritize tasks with smaller row number in addition to priority 2. We also enforces potrf>trsm>gemm
 */

enum PrioKind { no = 0, row = 1, cp = 2, cp_row = 3};

void cholesky3d(int n_threads, int verb, int block_size, int num_blocks, int npcols, int nprows, PrioKind prio, int test, int log, int debug)
{
	const int rank = comm_rank();
	const int n_ranks = comm_size();
	std::atomic<long long int> potrf_us_t(0);
	std::atomic<long long int> trsm_us_t(0);
	std::atomic<long long int> gemm_us_t(0);
	std::atomic<long long int> accu_us_t(0);

	int q = static_cast<int>(cbrt(n_ranks));
	if (q * q * q != n_ranks)
	{
		if (rank == 0)
		{
			cerr << "Number of processes must be a perfect cube." << endl;
		}
		MPI_Finalize();
		exit(1);
	}
	
	int3 rank_3d;
	int2 rank_2d;
	rank_3d[0] = rank / (q * q);
	rank_3d[1] = (rank % (q * q)) / q;
	rank_3d[2] = (rank % (q * q)) % q;
	rank_2d[0] = rank % nprows;
	rank_2d[1] = rank / nprows;

	// Number of tasks
	int n_tasks_per_rank = 2;
	struct acc_data {
		vector<std::unique_ptr<MatrixXd>> to_accumulate; // to_accumulate[k] holds matrix result of gemm(k,i,j)
	};

	auto potf_block_2_prio = [&](int j) {
		if (prio == PrioKind::cp_row) {
			return (double)(9 * (num_blocks - j) - 1) + 18 * num_blocks * num_blocks;
		}
		else if(prio == PrioKind::cp) {
			return (double)(9 * (num_blocks - j) - 1);
		}
		else if(prio == PrioKind::row) {
			return 3.0 * (double)(num_blocks - j);
		}
		else {
			return 3.0;
		}
	};

	auto trsm_block_2_prio = [&](int2 ij) {
		if (prio == PrioKind::cp_row) {
			return (double)((num_blocks - ij[0]) + num_blocks * (9.0 * num_blocks - 9.0 * ij[1] - 2.0) + 9 * num_blocks * num_blocks);
		}
		else if(prio == PrioKind::cp) {
			return (double)(9 * (num_blocks - ij[1]) - 2);
		}
		else if(prio == PrioKind::row) {
			return 2.0 * (double)(num_blocks - ij[0]);
		}
		else {
			return 2.0;
		}
	};

	auto gemm_block_2_prio = [&](int3 kij) {
		if (prio == PrioKind::cp_row) {
			return (double)(num_blocks - kij[1]) + num_blocks * (9.0 * num_blocks - 3.0 * kij[2] - 6.0 * (kij[0] / q) - 2.0);
		}
		else if(prio == PrioKind::cp) {
			return (double)(9 * num_blocks - 9 * kij[2] - 2);
		}
		else if(prio == PrioKind::row) {
			return (double)(num_blocks - kij[1]);
		}
		else {
			return 1.0;
		}
	};

	std::vector<acc_data> gemm_results(num_blocks*num_blocks);
	auto val = [&](int i, int j) { return 1/(double)((i - j)*(i - j) + 1); };
	auto rank3d21 = [&](int i, int j, int k) { return ((j % q) * q + k % q) + (i % q) * q * q;};
	auto rank2d21 = [&](int i, int j) { return (j % npcols) * nprows + (i % nprows);};
	auto rank1d21 = [&](int j) { return j % n_ranks; };
	vector<unique_ptr<MatrixXd>> blocks(num_blocks*num_blocks);

	auto bloc_2_rank = [&](int i, int j) {
		int r = (j % npcols) * nprows + (i % nprows);
		assert(r >= 0 && r < n_ranks);
		return r;
	};
 
	auto block_2_thread = [&](int i, int j) {
		int ii = i / nprows;
		int jj = j / npcols;
		int num_blocksit = num_blocks / nprows;
		return (ii + jj * num_blocksit) % n_threads;
	};
	{
		Eigen::MatrixXd A = Eigen::MatrixXd::Identity(256,256);
		Eigen::MatrixXd B = Eigen::MatrixXd::Identity(256,256);
		Eigen::MatrixXd C = Eigen::MatrixXd::Identity(256,256);
		for(int i = 0; i < 10; i++) {
			cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 256, 256, 256, 1.0, A.data(), 256, B.data(), 256, 1.0, C.data(), 256);
		}
	}
	for (int ii=0; ii<num_blocks; ii++) {
		for (int jj=0; jj<num_blocks; jj++) {
			auto val_loc = [&](int i, int j) { return val(ii*block_size+i,jj*block_size+j); };
			int dest = (ii == jj) ? rank1d21(ii) : rank2d21(ii,jj);
			if(dest == rank) {
				blocks[ii+jj*num_blocks]=make_unique<MatrixXd>(block_size, block_size);
				*blocks[ii+jj*num_blocks]=MatrixXd::NullaryExpr(block_size, block_size, val_loc);
				gemm_results[ii+jj*num_blocks].to_accumulate= vector<std::unique_ptr<MatrixXd>>(q);
				for (int ll=0; ll<q; ll++) {
					gemm_results[ii+jj*num_blocks].to_accumulate[ll]=make_unique<MatrixXd>(block_size, block_size);
				}
			} 
			else if (((ii % q) == rank_3d[0]) && ((jj % q) == rank_3d[1])) {
				blocks[ii+jj*num_blocks]=make_unique<MatrixXd>(block_size, block_size);
				*blocks[ii+jj*num_blocks]=MatrixXd::Zero(block_size, block_size);
			} 
			else {
				blocks[ii+jj*num_blocks]=make_unique<MatrixXd>(block_size, block_size);
			}
		}
	}
	// Initialize the communicator structure
	Communicator comm(MPI_COMM_WORLD, verb);
	// Initialize the runtime structures
	Threadpool tp(n_threads, &comm, verb, "WkTuto_" + to_string(rank) + "_");
	Taskflow<int> potrf(&tp, verb);
	Taskflow<int2> trsm(&tp, verb);
	Taskflow<int3> gemm(&tp, verb);
	Taskflow<int3> accu(&tp, verb);
	DepsLogger dlog(1000000);
	Logger ttorlog(1000000);
	if (log)  {
		tp.set_logger(&ttorlog);
		comm.set_logger(&ttorlog);
	}
	// Create active message
	auto am_trsm = comm.make_large_active_msg( 
			[&](int& j) {
				//int offset = ((j + 1) / nprows + (((j + 1) % nprows) > rank_2d[0])) * nprows + rank_2d[0];
				int offset = (nprows + rank_2d[0] - j % nprows) % nprows + j;
				for(int i = offset; i < num_blocks; i = i + nprows) {
					if (debug) printf("Fulfilling trsm (%d, %d) on rank (%d, %d)\n", i, j, rank_2d[0], rank_2d[1]);
					trsm.fulfill_promise({i,j});
				}
			},
			[&](int& j){
				return blocks[j+j*num_blocks]->data();
			},
			[&](int& j){
				return;
			});

	auto am_gemm = comm.make_large_active_msg(
		[&](int& i, int& k) {
			assert(k % q == rank_3d[2]);
			//int offset_c = ((k + 1) / q + (((k + 1) % q) > rank_3d[1])) * q + rank_3d[1]; 
			int offset_c = (q + rank_3d[1] - k % q) % q + k;
			if (i % q == rank_3d[0]) {
				for(int j = offset_c; j < i; j = j + q) {
					if (debug) printf("TRSM (%d, %d) Fulfilling gemm (%d, %d, %d) on rank (%d, %d, %d)\n", i, k, k, i, j, rank_3d[2], rank_3d[0], rank_3d[1]);
					gemm.fulfill_promise({k,i,j});
				}
			}
			//int offset_r = (i / q + ((i % q) > rank_3d[0])) * q + rank_3d[0];
			int offset_r = (q + rank_3d[0] - i % q) % q + i;
			if (i % q == rank_3d[1]) {
				for(int j = offset_r; j < num_blocks; j = j + q) {
					if (debug) printf("TRSM (%d, %d) Fulfilling gemm (%d, %d, %d) on rank (%d, %d, %d)\n", i,k, k, j, i, rank_3d[2], rank_3d[0], rank_3d[1]);      
					gemm.fulfill_promise({k,j,i});
				}
			}
		},
		[&](int& i, int& k) {
			return blocks[i+k*num_blocks]->data();
		},
		[&](int& i, int& k) {
			return;
		});

	potrf.set_task([&](int j) {
			assert(rank1d21(j) == rank);
			timer t1 = wctime();
			LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', block_size, blocks[j+j*num_blocks]->data(), block_size);
			timer t2 = wctime();
			potrf_us_t += 1e6 * elapsed(t1, t2);
			if (j == num_blocks - 1) printf("POTRF %d finished \n", j);
			if (debug) printf("Running POTRF %d on rank %d\n", j, rank);
		})
		.set_fulfill([&](int j) { 
			for (int p = 0; p < nprows; p++) 
			{   
				int r = rank2d21(p,j);
				if (rank == r) {
					//int offset = ((j + 1) / nprows + ((j + 1) % nprows) / (rank_2d[0] + 1)) * nprows + rank_2d[0];
					int offset = (nprows + rank_2d[0] - j % nprows) % nprows + j;
					for(int i = offset; i < num_blocks; i = i + nprows) {
						trsm.fulfill_promise({i,j});
					}
				}
				else {
					auto Ljjv = view<double>(blocks[j+j*num_blocks]->data(), block_size*block_size);
					am_trsm->send_large(r, Ljjv, j);
				}
			}
		})
		.set_indegree([&](int j) {
			if (j==0) {
				return 1;
			}
			else if (j < q) {
				return j;
			}
			else {
				return q;
			}
		})
		.set_mapping([&](int j) {
			return block_2_thread(j,j);
		})
		.set_binding([&](int j) {
			return false;

		})        
		.set_priority(potf_block_2_prio)
		.set_name([&](int j) { 
			return "POTRF" + to_string(j) + "_" + to_string(rank);
		});

	trsm.set_task([&](int2 ij) {
			int i=ij[0];
			int j=ij[1];
			assert(rank2d21(i,j) == rank);
			timer t1 = wctime();
			cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit, block_size, block_size, 1.0, blocks[j + j * num_blocks]->data(),block_size, blocks[i + j * num_blocks]->data(), block_size);
			timer t2 = wctime();
			trsm_us_t += 1e6 * elapsed(t1, t2);
			if (debug) printf("Running trsm (%d, %d) on rank %d, %d\n", i, j, rank_2d[0], rank_2d[1]);
		})
		.set_fulfill([&](int2 ij) {
			int i=ij[0];
			int j=ij[1]; 
			for (int ri = 0; ri < q; ri++)  {
				for (int rj = 0; rj < q; rj++) {
					int r = rank3d21(ri, rj, j);
					if (r == rank) {
						//int offset_c = ((j + 1) / q + (((j + 1) % q) > rank_3d[1])) * q + rank_3d[1];
						int offset_c = (q + rank_3d[1] - k % q) % q + k;
						if (i % q == rank_3d[0]) {
							for(int k = offset_c; k < i; k = k + q) {
								gemm.fulfill_promise({j,i,k});
							}
						}
						//int offset_r = (i / q + ((i % q) > rank_3d[0])) * q + rank_3d[0];
						int offset_r = (q + rank_3d[0] - i % q) % q + i;
						if (i % q == rank_3d[1]) {
							for(int k = offset_r; k < num_blocks; k = k + q) {
								gemm.fulfill_promise({j,k,i});
							} 
						}
					}
					else {
						auto Lijv = view<double>(blocks[i + j * num_blocks]->data(), block_size*block_size);
						am_gemm->send_large(r, Lijv, i, j);
					} 
				}
			}            
		})
		.set_indegree([&](int2 ij) {
			int i=ij[0];
			int j=ij[1];
			return (j < q) ? j + 1 : q + 1;
		})
		.set_mapping([&](int2 ij) {
			int i=ij[0];
			int j=ij[1];
			return block_2_thread(i,j);
		})
		.set_binding([&](int2 ij) {
			int i=ij[0];
			int j=ij[1];
			return false;

		})
		.set_priority(trsm_block_2_prio)
		.set_name([&](int2 ij) { 
			int i=ij[0];
			int j=ij[1];
			return "TRSM" + to_string(j) + "_" + to_string(i) + "_" +to_string(rank);
		});
   
	auto am_accu = comm.make_large_active_msg(
		[&](int& i, int& j, int& from) {
		accu.fulfill_promise({from, i, j});
		},
		[&](int& i, int& j, int& from){
			return gemm_results[i+j*num_blocks].to_accumulate[from]->data();
		},
		[&](int& i, int& j, int& from){
			return; 
		});
	
	gemm.set_task([&](int3 kij) {
			int k=kij[0];
			int i=kij[1];
			int j=kij[2]; 
			assert(rank3d21(i,j,k) == rank);
			timer t1 = wctime();           
			if (i==j) { 
				cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans, block_size, block_size, -1.0, blocks[i+k*num_blocks]->data(), block_size, 1.0, blocks[i+j*num_blocks]->data(), block_size);
			}
			else {
				cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, block_size, block_size, block_size, -1.0,blocks[i+k*num_blocks]->data(), block_size, blocks[j+k*num_blocks]->data(), block_size, 1.0, blocks[i+j*num_blocks]->data(), block_size);
			}
			timer t2 = wctime();
			if (debug) printf("Running gemm (%d, %d, %d) on rank %d, %d, %d\n", k, i, j, rank_3d[2], rank_3d[0], rank_3d[1]);
			gemm_us_t += 1;
		})
		.set_fulfill([&](int3 kij) { 
			int k=kij[0];
			int i=kij[1];
			int j=kij[2];
			if (k+q<=j-1) {
				gemm.fulfill_promise({k+q, i, j});
			}
			else {
				int dest = (i == j) ? rank1d21(i) : rank2d21(i, j);
				if (dest == rank) {
					if (debug) printf("gemm (%d, %d, %d) fulfilling accumu (%d, %d, %d) on rank %d, %d, %d\n", k, i, j, rank_3d[2], i, j, rank_3d[2], rank_3d[0], rank_3d[1]);
					accu.fulfill_promise({rank_3d[2], i, j});
				}
				else {
					int kk = rank_3d[2];
					auto Lij = view<double>(blocks[i+j*num_blocks]->data(), block_size*block_size);
					if (debug) printf("gemm (%d, %d, %d) Sending accumu (%d, %d, %d) to rank %d, %d\n", k, i, j, rank_3d[2], i, j, dest % nprows, dest / nprows);
					am_accu->send_large(dest, Lij, i, j, kk);
				}
			}
		})
		.set_indegree([&](int3 kij) {
			int k=kij[0];
			int i=kij[1];
			int j=kij[2];
			return 3 - (k/q == 0) - (i == j);
		})
		.set_mapping([&](int3 kij) {
			int k=kij[0];
			int i=kij[1];
			int j=kij[2];
			return block_2_thread(i,j);
		})
		.set_binding([&](int3 kij) {
			return false;

		})
		.set_priority(gemm_block_2_prio)
		.set_name([&](int3 kij) { // This is just for debugging and profiling
			int k=kij[0];
			int i=kij[1];
			int j=kij[2];
			return "gemm" + to_string(k) + "_" + to_string(i)+"_"+to_string(j)+"_"+to_string(comm_rank());
		});

	accu.set_task([&](int3 kij) {
			int k=kij[0]; // Step (gemm's pivot)
			int i=kij[1]; // Row
			int j=kij[2]; // Col
			int dest = (i == j) ? rank1d21(i) : rank2d21(i,j);
			assert(dest == rank);
			assert(j <= i);
			if (debug) printf("Running accumu (%d, %d, %d) on rank %d, %d\n", k, i, j, rank % nprows, rank / nprows);
			{
				timer t_ = wctime();
				*blocks[i+j*num_blocks] += (*gemm_results[i+j*num_blocks].to_accumulate[k]);
				timer t__ = wctime();
				accu_us_t += 1e6 * elapsed(t_, t__);
			}
		})
		.set_fulfill([&](int3 kij) {
			int k=kij[0];
			int i=kij[1];
			int j=kij[2];
			assert(j <= i);
			if(i == j) {
				potrf.fulfill_promise(i);
			} else {
				trsm.fulfill_promise({i,j});
			}
		})
		.set_indegree([&](int3 kij) {
			return 1;
		})
		.set_mapping([&](int3 kij) {
			int k=kij[0];
			int i=kij[1];
			int j=kij[2];
			return block_2_thread(i,j);
		})
		.set_priority(gemm_block_2_prio)
		.set_binding([&](int3 kij) {
			return true; // IMPORTANT
		})
		.set_name([&](int3 kij) { // This is just for debugging and profiling
			int k=kij[0];
			int i=kij[1];
			int j=kij[2];
			return "accumu" + to_string(k) + "_" + to_string(i)+"_"+to_string(j)+"_"+to_string(comm_rank());
		});

	MPI_Barrier(MPI_COMM_WORLD);
	timer t0 = wctime();
	if (rank == 0){
		potrf.fulfill_promise(0);
	}
	tp.join();
	MPI_Barrier(MPI_COMM_WORLD);
	timer t1 = wctime();
	MPI_Status status;
	if (rank==0) {
		cout<<"3D, Number of ranks "<<n_ranks<<", block_size "<<block_size<<", num_blocks "<<num_blocks<<", n_threads "<<n_threads<<", Priority "<<prio<<", Elapsed time: "<<elapsed(t0,t1)<<endl;
	}
	if (test)   {
		MatrixXd A;
		A = MatrixXd::NullaryExpr(block_size*num_blocks,block_size*num_blocks, val);
		MatrixXd L = A;
		for (int ii=0; ii<num_blocks; ii++) {
			for (int jj=0; jj<num_blocks; jj++) {
				if (jj<=ii)  {
					if (rank==0 && rank!=rank2d21(ii, jj)) {
						blocks[ii+jj*num_blocks]=make_unique<MatrixXd>(block_size,block_size);
						MPI_Recv(blocks[ii+jj*num_blocks]->data(), block_size*block_size, MPI_DOUBLE, rank2d21(ii, jj), 0, MPI_COMM_WORLD, &status);
					}
					else if (rank==rank2d21(ii, jj) && rank != 0) {
						MPI_Send(blocks[ii+jj*num_blocks]->data(), block_size*block_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
					}
				}
			}
		}
		if (rank == 0)  {
			for (int ii=0; ii<num_blocks; ii++) {
				for (int jj=0; jj<num_blocks; jj++) {
					if (jj<=ii)  {
						L.block(ii*block_size,jj*block_size,block_size,block_size)=*blocks[ii+jj*num_blocks];
					}
				}
			}
		}
		if (rank == 0) {
			auto L1=L.triangularView<Lower>();
			VectorXd x = VectorXd::Random(block_size * num_blocks);
			VectorXd b = A*x;
			L1.solveInPlace(b);
			L1.transpose().solveInPlace(b);
			double error = (b - x).norm() / x.norm();
			cout << "Error solve: " << error << endl;
		}
	}
	if (log)  {
		std::ofstream logfile;
		string filename = "ttor_3Dcholesky_Priority_"+to_string(block_size)+"_"+to_string(num_blocks)+"_"+ to_string(n_threads)+"_"+ to_string(n_ranks)+"_"+ to_string(prio)+".log."+to_string(rank);
		logfile.open(filename);
		logfile << ttorlog;
		logfile.close();
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
	int npcols = 1;
	int nprows = ttor::comm_size();
	PrioKind prio = PrioKind::no;
	int test = 0;
	int log = 0;
	int debug = 0;
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
	if (argc >= 5) {
		n_threads=atoi(argv[3]);
		assert(n_threads > 0);
		verb=atoi(argv[4]);
		assert(verb >= 0);
	}
	if (argc >= 7) {
		npcols=atoi(argv[5]);
		assert(npcols > 0);
		nprows=atoi(argv[6]);
		assert(nprows > 0);
	}
	if (argc >= 8) {
		prio=(PrioKind)atoi(argv[7]);
		assert(prio >= 0 && prio <4);
	}
	if (argc >= 9) {
		test=atoi(argv[8]);
		assert(test == 0 || test == 1);
	}
	if (argc >= 10) {
		log = atoi(argv[9]);
		assert(log == 0 || log == 1);
	}
	if (argc >= 11) {
		debug = atoi(argv[10]);
		assert(debug == 0 || debug == 1);
	}
	if(comm_rank() == 0) printf("Usage: ./3d_cholesky block_size num_blocks n_threads verb nprows npcols priority test log debug\n");
	cholesky3d(n_threads, verb, block_size, num_blocks, npcols, nprows, prio, test, log, debug);  
	MPI_Finalize();
}
