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


using namespace std;
using namespace Eigen;
using namespace ttor;

using namespace std;
using namespace ttor;


typedef array<int, 2> int2;
typedef array<int, 3> int3;





//Test Test2
void tuto_1(int n_threads, int verb, int n, int nb)
{
    int rank = 0;
    double gemm_t=0;
    double potrf_t=0;
    double trsm_t=0;
    double syrk_t=0;


    // Number of tasks
    int n_tasks_per_rank = 2;


    auto val = [&](int i, int j) { return 1/(float)((i-j)*(i-j)+1); };
    MatrixXd A;
    A = MatrixXd::NullaryExpr(n*nb,n*nb, val);
    MatrixXd L = A;
    vector<unique_ptr<MatrixXd>> blocs(nb*nb);
    for (int ii=0; ii<nb; ii++) {
        for (int jj=0; jj<nb; jj++) {
            blocs[ii+jj*nb]=make_unique<MatrixXd>(n,n);

            *blocs[ii+jj*nb]=L.block(ii*n,jj*n,n,n);
        }
    }
    //cout<<A->rows()<<"\n";
    //LLT<MatrixXd> Test(A);
    //MatrixXd LR=Test.matrixL();

    // Outgoing dependencies for each task


    // Map tasks to rank
    auto task_2_rank = [&](int k) {
        return k / n_tasks_per_rank;
    };

    // Initialize the runtime structures
    Threadpool tp(n_threads, verb, "WkTuto_" + to_string(rank) + "_");
    Taskflow<int> potrf(&tp, verb);
    Taskflow<int2> trsm(&tp, verb);
    Taskflow<int3> gemm(&tp, verb);

    // Create active message


    // Define the task flow
    potrf.set_task([&](int k) {
          timer t0 = wctime();
          LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', n, blocs[k+k*nb]->data(), n);
          timer t1 = wctime();
          potrf_t+=elapsed(t0,t1);

      })
        .set_fulfill([&](int k) {
            for (int p = k+1; p<nb; p++) // Looping through all outgoing dependency edges
            {
                int dest = task_2_rank(p); // defined above

                trsm.fulfill_promise({k,p}, 5.0);

            }
        })
        .set_indegree([&](int k) {
            return 1;
        })
        .set_mapping([&](int k) {

            return (k % n_threads);
        })
        .set_binding([&](int k) {
            return false;

        })
        .set_name([&](int k) { // This is just for debugging and profiling
            return "POTRF" + to_string(k) + "_" + to_string(rank);
        });



    trsm.set_task([&](int2 ki) {
        int k=ki[0];
        int i=ki[1];
        //cout<<L.block(i*n,k*n,n,n)<<endl;
        //MatrixXd temp=L.block(i*n,k*n,n,n);
        //MatrixXd temp2=L.block(k*n,k*n,n,n);
        //auto T=L.block(k*n,k*n,n,n).triangularView<Lower>().transpose().solve<OnTheRight>(L.block(i*n,k*n,n,n));
        timer t0 = wctime();
        cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit, n, n, 1.0, blocs[k+k*nb]->data(),n, 
blocs[i+k*nb]->data(), n);
        timer t1 = wctime();
        trsm_t+=elapsed(t0,t1);
        //L.block(i*n,k*n,n,n)=temp;
        //cout<<L.block(k*n,k*n,n,n)<<"\n\n";
        //cout << "LAPACK: \n";
        //cout << temp*L.block(k*n,k*n,n,n).transpose() << "\n\n";
        //cout << "Eigen: \n";
        //cout << T*L.block(k*n,k*n,n,n).transpose()<< "\n\n";
        //L.block(i*n,k*n,n,n)=T;
        //MatrixXd Temp=L.block(i*n, k*n, n, n);
        //cout<<(LR.block(i*n, k*n, n, n)-L.block(i*n, k*n, n, n)).norm()<<endl;
        //cout<<Temp(0,0)<<endl;
        //cout<<LR(i,k)<<endl;
      })
        .set_fulfill([&](int2 ki) {
            int k=ki[0];
            int i=ki[1];
            for (int j=k+1; j<nb;j++) // Looping through all outgoing dependency edges
            {

                if (j<i) {
                    gemm.fulfill_promise({k,i,j}, 5.0);
                }
                else {
                    gemm.fulfill_promise({k,j,i}, 5.0);
                }

            }
        })
        .set_indegree([&](int2 ki) {
            int k=ki[0];
            int i=ki[1];
            if (k==0) {
                return 1;
            }
            else {
                return 2;
            }
        })
        .set_mapping([&](int2 ki) {
            int k=ki[0];
            int i=ki[1];

            return ((k*n+i) % n_threads);
        })
        .set_binding([&](int2 ki) {
            int k=ki[0];
            int i=ki[1];
            return false;

        })
        .set_name([&](int2 ki) { // This is just for debugging and profiling
            int k=ki[0];
            int i=ki[1];
            return "TRSM" + to_string(k) + "_" + to_string(i) + "_" +to_string(rank);
        });


    gemm.set_task([&](int3 kij) {
            int k=kij[0];
            int i=kij[1];
            int j=kij[2];
            //L.block(i*n, j*n, n, n)-=L.block(i*n, k*n, n, n)*L.block(j*n, k*n, n, n).transpose();
            //MatrixXd blocij=L.block(i*n, j*n, n, n);
            //MatrixXd blocik=L.block(i*n, k*n, n, n);
            //MatrixXd blocjk=L.block(j*n, k*n, n, n);
            if (i==j) {
                timer t0 = wctime();
                cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans, n, n, -1.0, blocs[i+k*nb]->data(), n, 1.0, 
blocs[i+j*nb]->data(), n);
                timer t1 = wctime();
                syrk_t+=elapsed(t0,t1);
            }
            else {
                timer t0 = wctime();
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, n, n, n, -1.0,blocs[i+k*nb]->data(), n, 
blocs[j+k*nb]->transpose().data(), n, 1.0, blocs[i+j*nb]->data(), n);
                timer t1 = wctime();
                gemm_t+=elapsed(t0,t1);
            }
            
            
            
            //L.block(i*n, j*n, n, n)=blocij;
            //cout<<Temp(0,0)<<endl;
      })
        .set_fulfill([&](int3 kij) {
            int k=kij[0];
            int i=kij[1];
            int j=kij[2];
            if (k<j-1) {
                gemm.fulfill_promise({k+1, i, j}, 5.0);
            }
            else {
                if (i==j) {
                    potrf.fulfill_promise(i, 5.0);
                }
                else {
                    trsm.fulfill_promise({j,i}, 5.0);
                }
            }
            

        })
        .set_indegree([&](int3 kij) {
            int k=kij[0];
            int i=kij[1];
            int j=kij[2];
            int t=3;
            if (k==0) {
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

            return ((k*n*n+i+j*n)  % n_threads);
        })
        .set_binding([&](int3 kij) {
            return false;

        })
        .set_name([&](int3 kij) { // This is just for debugging and profiling
            int k=kij[0];
            int i=kij[1];
            int j=kij[2];
            return "GEMM" + to_string(k) + "_" + to_string(i)+"_"+to_string(j);
        });


    

    // Seed initial tasks
    potrf.fulfill_promise(0, 5.0);

    // Other ranks do nothing
    // Run until completion
    timer t0 = wctime();
    tp.join();
    timer t1 = wctime();
    for (int ii=0; ii<nb; ii++) {
        for (int jj=0; jj<nb; jj++) {
            L.block(ii*n,jj*n,n,n)=*blocs[ii+jj*nb];
        }
    }
    auto L1=L.triangularView<Lower>();
    cout<<"Elapsed time: "<<elapsed(t0,t1)<<endl;

    VectorXd x = VectorXd::Random(n * nb);
    VectorXd b = A*x;
    VectorXd bref = b;
    L1.solveInPlace(b);
    L1.transpose().solveInPlace(b);
    double error = (b - x).norm() / x.norm();
    cout << "Error solve: " << error << endl;
    cout << "POTRF: " << potrf_t << endl;
    cout << "TRSM: " << trsm_t << endl;
    cout << "SYRK: " << syrk_t << endl;
    cout << "GEMM: " << gemm_t << endl;
    //cout<<"LLT Error: "<<(A-L*L.transpose()).norm()/A.norm()<<"\n";
    //cout<<"LLT Error for Eigen : "<<(A-L1*L1.transpose()).norm()/A.norm()<<"\n";
    //cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, L.rows(), L.rows(), L.cols(), -1.0, L.data(), L.rows(), L.data(), 
L.rows(), 0.0, A.data(), L.rows());
    //cout<<"LLT Error GT: "<<(A-LR*LR.transpose()).norm()/A.norm()<<"\n";
}

int main(int argc, char **argv)
{

    int n_threads = 2;
    int verb = 0; // Can be changed to vary the verbosity of the messages
    int n=1;
    int nb=2;


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


    tuto_1(n_threads, verb, n, nb);

}
