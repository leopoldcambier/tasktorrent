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


    timer t0 = wctime();
    LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', n, A.data(), n);
    timer t1 = wctime();
    for (int ii=0; ii<nb; ii++) {
        for (int jj=0; jj<nb; jj++) {
            L.block(ii*n,jj*n,n,n)=*blocs[ii+jj*nb];
        }
    }
    auto L1=L.triangularView<Lower>();
    cout<<"Elapsed time: "<<elapsed(t0,t1)<<endl;


    /*
    std::ofstream logfile;
    std::string filename = "ttor_shared_"+ to_string(n)+"_"+to_string(nb)+".log."+to_string(rank);
    logfile.open(filename);
    logfile << logger;
    logfile.close();
    */



    VectorXd x = VectorXd::Random(n * nb);
    VectorXd b = A*x;
    VectorXd bref = b;
    L1.solveInPlace(b);
    L1.transpose().solveInPlace(b);
    double error = (b - x).norm() / x.norm();
    cout << "Error solve: " << error << endl;

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