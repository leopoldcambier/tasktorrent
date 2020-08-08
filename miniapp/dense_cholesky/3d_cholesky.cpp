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

void cholesky3d(int n_threads, int verb, int n, int nb, int n_col, int n_row, int priority, int test, int LOG, int tm, int debug)
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
    rank_2d[0] = rank % n_row;
    rank_2d[1] = rank / n_row;

    // Number of tasks
    int n_tasks_per_rank = 2;

    
    struct acc_data {
        vector<std::unique_ptr<MatrixXd>> to_accumulate; // to_accumulate[k] holds matrix result of gemm(k,i,j)
    };





    std::vector<acc_data> gemm_results(nb*nb);

    auto val = [&](int i, int j) { return 1/(double)((i-j)*(i-j)+1); };
    auto rank3d21 = [&](int i, int j, int k) { return ((j % q) * q + k % q) + (i % q) * q * q;};
    auto rank2d21 = [&](int i, int j) { return (j % n_col) * n_row + i % n_row;};
    auto rank1d21 = [&](int k) { return k % n_ranks; };
    vector<unique_ptr<MatrixXd>> blocs(nb*nb);

    auto bloc_2_rank = [&](int i, int j) {
        int r = (j % n_col) * n_row + (i % n_row);
        assert(r >= 0 && r < n_ranks);
        return r;
    };

    auto block_2_thread = [&](int i, int j) {
        int ii = i / n_row;
        int jj = j / n_col;
        int num_blocksit = nb / n_row;
        return (ii + jj * num_blocksit) % n_threads;
    };


    for (int ii=0; ii<nb; ii++) {
        for (int jj=0; jj<nb; jj++) {
            auto val_loc = [&](int i, int j) { return val(ii*n+i,jj*n+j); };
            if(rank2d21(ii,jj) == rank) {
                blocs[ii+jj*nb]=make_unique<MatrixXd>(n, n);
                *blocs[ii+jj*nb]=MatrixXd::NullaryExpr(n, n, val_loc);
                gemm_results[ii+jj*nb].to_accumulate= vector<std::unique_ptr<MatrixXd>>(q);
                for (int ll=0; ll<q; ll++) {
                    gemm_results[ii+jj*nb].to_accumulate[ll]=make_unique<MatrixXd>(n, n);
                }
            } 

            else if (((ii % q) == rank_3d[0]) && ((jj % q) == rank_3d[1])) {
                blocs[ii+jj*nb]=make_unique<MatrixXd>(n, n);
                *blocs[ii+jj*nb]=MatrixXd::Zero(n, n);
            }
            
            else {
                blocs[ii+jj*nb]=make_unique<MatrixXd>(n, n);
                //blocs[ii+jj*nb]=make_unique<MatrixXd>(n, n);
                //*blocs[ii+jj*nb]=MatrixXd::Zero(n, n);
            }
        }
    }



    // Map tasks to rank
    

    // Initialize the communicator structure
    Communicator comm(MPI_COMM_WORLD, verb);

    // Initialize the runtime structures
    Threadpool tp(n_threads, &comm, verb, "WkTuto_" + to_string(rank) + "_");
    Taskflow<int> potrf(&tp, verb);
    Taskflow<int2> trsm(&tp, verb);
    Taskflow<int3> gemm(&tp, verb);
    Taskflow<int3> accu(&tp, verb);

    
    DepsLogger dlog(1000000);
    Logger log(1000000);

    if (LOG)  {
        tp.set_logger(&log);
        comm.set_logger(&log);
    }
    

    // Create active message
    auto am_trsm = comm.make_large_active_msg( 
            [&](int& j) {
                int offset = (j / n_row + (j % n_row) / (rank_2d[0] + 1)) * n_row + rank_2d[0] + 1;
                for(int i = offset; i < nb; i = i + n_row) {
                    trsm.fulfill_promise({j,i});
                }
            },
            [&](int& j){
                return blocs[j+j*nb]->data();
            },
            [&](int& j){
                return;
            });

        // Sends a panel bloc and trigger multiple gemms 
    auto am_gemm = comm.make_large_active_msg(
        [&](int& i, int& k) {
            assert(k % q == rank_3d[2]);
            int offset_c = (k / n_col + (k % n_col) / (rank_3d[1] + 1)) * n_col + rank_3d[1] + 1; 
            for(int j = offset_c; j < i; j = j + n_col ) {
                gemm.fulfill_promise({k,i,j});
            }
	    int offset_r = (i / n_row + (i % n_row) / (rank_3d[0] + 1)) * n_row + rank_3d[0];
            for(int j = offset_r; j < nb; j = j + n_row) {
                gemm.fulfill_promise({k,j,i});
            }
        },
        [&](int& i, int& k) {
            return blocs[i+k*nb]->data();
        },
        [&](int& i, int& k) {
            return;
        });

    // Define the task flow
    potrf.set_task([&](int j) {
          timer t1 = wctime();
          LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', n, blocs[j+j*nb]->data(), n);
          timer t2 = wctime();
          potrf_us_t += 1e6 * elapsed(t1, t2);
          //potrf_us_t += 1;
          if (debug) printf("Running POTRF %d on rank %d, %d, %d\n", j, rank_3d[0], rank_3d[1], rank_3d[2]);
      })
        .set_fulfill([&](int j) { 
            for (int r = 0; r < n_ranks; r++) // Looping through all outgoing dependency edges
            {
                if (rank == r) {
                   int offset = (j / n_row + (j % n_row) / (rank_2d[0] + 1)) * n_row + rank_2d[0] + 1;
                   for(int i = offset; i < nb; i = i + n_row) {
                      trsm.fulfill_promise({j,i});
                   }
                }
                else {
                    auto Ljjv = view<double>(blocs[j+j*nb]->data(), n*n);
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
            if (tm) {
                return block_2_thread(j,j);
            }
            return (j % n_threads);
        })
        .set_binding([&](int j) {
            return false;

        })        
        .set_priority([&](int j) {
            if (priority==-1) {
                return 1.0;
            }
            if (priority==0) {
                return 3.0;
            }
            if (priority==1) {
                return 9.0*(double)(nb-j)-1.0;
            }
            if (priority==2) {
                return 9.0*(double)(nb-j)-1.0+18*nb;
            }
            if (priority==3) {
                return 9.0*(double)(nb-j)-1.0+18*nb*nb;
            }
            else {
                return 3.0*(double)(nb-j);
            }
        })
        .set_name([&](int j) { // This is just for debugging and profiling
            return "POTRF" + to_string(j) + "_" + to_string(rank);
        });



    trsm.set_task([&](int2 ki) {
        int k=ki[0];
        int i=ki[1];
        timer t1 = wctime();
        cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit, n, n, 1.0, blocs[k+k*nb]->data(),n, blocs[i+k*nb]->data(), n);
        timer t2 = wctime();
        trsm_us_t += 1e6 * elapsed(t1, t2);
        //trsm_us_t += 1;
        if (debug) printf("Running TRSM (%d, %d) on rank %d, %d, %d\n", i, k, rank_3d[0], rank_3d[1], rank_3d[2]);
        //cout<<(*blocs[i+k*nb])<<"\n";

      })
        .set_fulfill([&](int2 ki) {
            int k=ki[0];
            int i=ki[1]; 
            for (int r = 0; r < n_ranks; r++)   // Looping through all outgoing dependency edges
            {
                if (r == rank) {
                  //if ((k % q) != rank_3d[2]) continue;
                  int offset_c = (k / n_col + (k % n_col) / (rank_3d[1] + 1)) * n_col + rank_3d[1] + 1;
                  for(int j = offset_c; j < i; j = j + n_col ) {
                    gemm.fulfill_promise({k,i,j});
                  }
                  int offset_r = (i / n_row + (i % n_row) / (rank_3d[0] + 1)) * n_row + rank_3d[0];
                  for(int j = offset_r; j < nb; j = j + n_row) {
                    gemm.fulfill_promise({k,j,i});
                  }
                }
                else {
                    //cout<<"Sending data from "<<rank<<" to "<<r<<"\n";
                    auto Lijv = view<double>(blocs[i+k*nb]->data(), n*n);
                    am_gemm->send_large(r, Lijv, i, k);
                }
            }            
        })
        .set_indegree([&](int2 ki) {
            int k=ki[0];
            int i=ki[1];
            if (k==0) {
                return 1;
            }
            else if (k < q) {
                return k+1;
            }
            else {
                return q+1;
            }
        })
        .set_mapping([&](int2 ki) {
            int k=ki[0];
            int i=ki[1];
            if (tm) {
                return block_2_thread(i,k);
            }
            return ((k*n+i) % n_threads);
        })
        .set_binding([&](int2 ki) {
            int k=ki[0];
            int i=ki[1];
            return false;

        })
        .set_priority([&](int2 ki) {
            int k=ki[0];
            int i=ki[1];
            if (priority==-1) {
                return 1.0;
            }
            if (priority==0) {
                return 2.0;
            }
            if (priority==1) {
                return 9.0*(double)(nb-k)-2.0;
            }
            if (priority==2) {
                return 9.0*(double)(nb-k)-2.0+9*nb;
            }
            if (priority==3) {
                return (double)(nb-i)+nb*(9.0*(double)(nb-k)-2.0)+9*nb*nb;
            }
            else {
                return 2.0*(double)(nb-i);
            }
        })
        .set_name([&](int2 ki) { // This is just for debugging and profiling
            int k=ki[0];
            int i=ki[1];
            return "TRSM" + to_string(k) + "_" + to_string(i) + "_" +to_string(rank);
        });
   
    auto am_accu = comm.make_large_active_msg(
        [&](int& i, int& j, int& from) {
        accu.fulfill_promise({from, i, j});
        },
        [&](int& i, int& j, int& from){
            return gemm_results[i+j*nb].to_accumulate[from]->data();
        },
        [&](int& i, int& j, int& from){
            return; 
        });
    



    gemm.set_task([&](int3 kij) {
            int k=kij[0];
            int i=kij[1];
            int j=kij[2]; 
            timer t1 = wctime();           
            if (i==j) { 
                cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans, n, n, -1.0, blocs[i+k*nb]->data(), n, 1.0, blocs[i+j*nb]->data(), n);
            }
            else {
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, n, n, n, -1.0,blocs[i+k*nb]->data(), n, blocs[j+k*nb]->data(), n, 1.0, blocs[i+j*nb]->data(), n);
            }
            timer t2 = wctime();
            if (debug) printf("Running GEMM (%d, %d, %d) on rank %d, %d, %d\n", i, j, k, rank_3d[0], rank_3d[1], rank_3d[2]);
            //gemm_us_t += 1e6 * elapsed(t1,t2);
            gemm_us_t += 1;
            

      })
        .set_fulfill([&](int3 kij) { 
            int k=kij[0];
            int i=kij[1];
            int j=kij[2];
            if (k+q<=j-1) {
                gemm.fulfill_promise({k+q, i, j});
                //printf("Gemm (%d, %d, %d) fulfilling Gemm (%d , %d, %d) on rank %d\n", k, i, j, k+1, i, j, comm_rank());
            }
            else {
                int dest = rank2d21(i, j);
                if (dest == rank) {
                    if (debug) printf("Gemm (%d, %d, %d) fulfilling ACCUMU (%d, %d, %d) on rank %d, %d, %d\n", k, i, j, rank_3d[2], i, j, rank_3d[0], rank_3d[1], rank_3d[2]);

                    accu.fulfill_promise({rank_3d[2], i, j});
                }

                else {
                    int kk = rank_3d[2];
                    auto Lij = view<double>(blocs[i+j*nb]->data(), n*n);
                    if (debug) printf("Gemm (%d, %d, %d) Sending ACCUMU (%d, %d, %d) to rank %d, %d\n", k, i, j, rank_3d[2], i, j, dest % n_row, dest / n_row);
                    am_accu->send_large(dest, Lij, i, j, kk);
                }
            }
            

        })
        .set_indegree([&](int3 kij) {
            int k=kij[0];
            int i=kij[1];
            int j=kij[2];
            int t=3;
            if ((k/q)==0) {
                t--;
            }
            if (i==j) {
                t--;
            }
            return t;
        })
        .set_mapping([&](int3 kij) {
            int k=kij[0];
            int i=kij[1];
            int j=kij[2];

            if (tm) {
                return block_2_thread(i,j);
            }
            return ((k*n*n+i+j*n)  % n_threads);
            //return block_2_thread(i,j);
        })
        .set_binding([&](int3 kij) {
            return false;

        })
        .set_priority([&](int3 kij) {
            int k=kij[0];
            int i=kij[1];
            int j=kij[2];
            if (priority==-1) {
                return 1.0;
            }
            if (priority==0) {
                return 1.0;
            }
            if (priority==1) {
                return 9.0*nb-3.0*j-6.0*k-2.0;
            }
            if (priority==2) {
                return 9.0*nb-3.0*j-6.0*k-2.0;
            }
            if (priority==3) {
                return (double)(nb-i)+nb*(9.0*nb-3.0*j-6.0*k-2.0);
            }
            else {
                return 1.0*(double)(nb-i);
            }

        })
        .set_name([&](int3 kij) { // This is just for debugging and profiling
            int k=kij[0];
            int i=kij[1];
            int j=kij[2];
            return "GEMM" + to_string(k) + "_" + to_string(i)+"_"+to_string(j)+"_"+to_string(comm_rank());
        });

    accu.set_task([&](int3 kij) {
            //assert(rank_3d21(kij[1],kij[2],kij[2]) == rank);
            int k=kij[0]; // Step (gemm's pivot)
            int i=kij[1]; // Row
            int j=kij[2]; // Col
            assert(rank2d21(i,j) == rank);
            assert(j <= i);
            if (debug) printf("Running ACCU (%d, %d, %d) on rank %d, %d\n", k, i, j, rank % n_row, rank / n_row);
            //assert(k < j);
            //std::unique_ptr<Eigen::MatrixXd> Atmp;
            {
                timer t_ = wctime();
                *blocs[i+j*nb] += (*gemm_results[i+j*nb].to_accumulate[k]);
                timer t__ = wctime();
                accu_us_t += 1e6 * elapsed(t_, t__);
            }
            //cout<<(*blocs[i+j*nb])<<"\n";
            //cout<<(*Atmp)<<"\n";
            //timer t_ = wctime();
            //*blocs[i+j*nb] += (*Atmp);
            //timer t__ = wctime();
            //accu_us_t += 1e6 * elapsed(t_, t__);
            //printf("Running ACCU (%d, %d, %d) on rank %d, %d\n", k, i, j, rank % n_row, rank / n_row);
        })
        .set_fulfill([&](int3 kij) {
            //assert(rank_3d21(kij[1],kij[2],kij[2]) == rank);
            int k=kij[0];
            int i=kij[1];
            int j=kij[2];
            assert(j <= i);
            if(i == j) {
                potrf.fulfill_promise(i);
            } else {
                trsm.fulfill_promise({j,i});
            }
        })
        .set_indegree([&](int3 kij) {

            //assert(rank_3d21(kij[1] % q, kij[2] % q,kij[2] % q) == rank);
            
            return 1;
        })
        .set_mapping([&](int3 kij) {
            int k=kij[0];
            int i=kij[1];
            int j=kij[2];
            if (tm) {
                return block_2_thread(i,j);
            }
            //return block_2_thread(i,j);
            return ((i+j*n)  % n_threads);
            //return ((k*n*n+i+j*n)  % n_threads);// IMPORTANT. Every (i,j) should map to a given fixed thread
        })
        .set_priority([&](int3 kij) {
            int k=kij[0];
            int i=kij[1];
            int j=kij[2];
            if (priority==-1) {
                return 1.0;
            }
            if (priority==0) {
                return 1.0;
            }
            if (priority==1) {
                return 9.0*nb-3.0*j-6.0*k-2.0;
            }
            if (priority==2) {
                return 9.0*nb-3.0*j-6.0*k-2.0;
            }
            if (priority==3) {
                return (double)(nb-i)+nb*(9.0*nb-3.0*j-6.0*k-2.0);
            }
            else {
                return 1.0*(double)(nb-i);
            }

        })
        .set_binding([&](int3 kij) {
            //assert(rank3d21(kij[1],kij[2],kij[2]) == rank);
            return true; // IMPORTANT
        })
        .set_name([&](int3 kij) { // This is just for debugging and profiling
            //assert(rank3d21(kij[1],kij[2],kij[2]) == rank);
            int k=kij[0];
            int i=kij[1];
            int j=kij[2];
            return "ACCUMU" + to_string(k) + "_" + to_string(i)+"_"+to_string(j)+"_"+to_string(comm_rank());
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
        cout<<"3D, Number of ranks "<<n_ranks<<", n "<<n<<", nb "<<nb<<", n_threads "<<n_threads<<", tm "<<tm<<", Priority "<<priority<<", Elapsed time: "<<elapsed(t0,t1)<<endl;
    }

    if (test)   {
        MatrixXd A;
        A = MatrixXd::NullaryExpr(n*nb,n*nb, val);
        MatrixXd L = A;

        for (int ii=0; ii<nb; ii++) {
            for (int jj=0; jj<nb; jj++) {
                if (jj<=ii)  {
                if (rank==0 && rank!=rank2d21(ii, jj)) {
                    blocs[ii+jj*nb]=make_unique<MatrixXd>(n,n);
                    MPI_Recv(blocs[ii+jj*nb]->data(), n*n, MPI_DOUBLE, rank2d21(ii, jj), 0, MPI_COMM_WORLD, &status);
                    }

                else if (rank==rank2d21(ii, jj) && rank != 0) {
                    MPI_Send(blocs[ii+jj*nb]->data(), n*n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                    }
                }
            }
        }
        
        if (rank == 0)  {
            for (int ii=0; ii<nb; ii++) {
                for (int jj=0; jj<nb; jj++) {
                    if (jj<=ii)  {
                        L.block(ii*n,jj*n,n,n)=*blocs[ii+jj*nb];
                    }
                }
            }
        }
        if (rank == 0) {
        auto L1=L.triangularView<Lower>();
        LLT<MatrixXd> lltOfA(A);
        MatrixXd TrueL= lltOfA.matrixL();
/*         if (rank==0) {
            cout << "True L:\n"<<TrueL<<"\n";
            cout << "L: \n"<<L<<"\n";
        } */
        VectorXd x = VectorXd::Random(n * nb);
        VectorXd b = A*x;
        VectorXd bref = b;
        L1.solveInPlace(b);
        L1.transpose().solveInPlace(b);
        double error = (b - x).norm() / x.norm();
        
            cout << "Error solve: " << error << endl;
        }
    }

    if (LOG)  {
        std::ofstream logfile;
        string filename = "ttor_3Dcholesky_Priority_"+to_string(n)+"_"+to_string(nb)+"_"+ to_string(n_threads)+"_"+ to_string(n_ranks)+"_"+ to_string(priority)+".log."+to_string(rank);
        logfile.open(filename);
        logfile << log;
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
    int n=1;
    int nb=2;
    int n_col=1;
    int n_row=1;
    int priority=0;
    int test=0;
    int log=0;
    int tm=0;
    int debug=0;

    if (argc >= 2)
    {
        n = atoi(argv[1]);
    }

    if (argc >= 3)
    {
        nb = atoi(argv[2]);
    }
    

    if (argc >= 5) {
        n_threads=atoi(argv[3]);
        verb=atoi(argv[4]);
    }

    if (argc >= 7) {
        n_col=atoi(argv[5]);
        n_row=atoi(argv[6]);
    }

    if (argc >= 8) {
        priority=atoi(argv[7]);
    }

    if (argc >= 9) {
        test=atoi(argv[8]);
    }

    if (argc >= 10) {
        log = atoi(argv[9]);
    }

    if (argc >= 11) {
        tm = atoi(argv[11]);
    }
    if (argc >= 12) {
        debug = atoi(argv[12]);
    }

    cholesky3d(n_threads, verb, n, nb, n_col, n_row, priority, test, log, tm, debug);  

    MPI_Finalize();
}
