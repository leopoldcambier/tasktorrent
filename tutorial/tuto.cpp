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

using namespace std;
using namespace Eigen;
using namespace ttor;

using namespace std;
using namespace ttor;


typedef array<int, 2> int2;
typedef array<int, 3> int3;

int n = 10;
int nb = 10;



//Test Test2
void tuto_1(int n_threads, int verb)
{
    const int rank = comm_rank();
    const int n_ranks = comm_size();

    if (n_ranks < 2)
    {
        printf("You need to run this code with at least 2 MPI processors\n");
        exit(0);
    }

    printf("Rank %d hello from %s\n", rank, processor_name().c_str());

    // Number of tasks
    int n_tasks_per_rank = 2;


    auto val = [&](int i, int j) {return i+j;};
    MatrixXd A = MatrixXd::NullaryExpr(n*nb,n*nb, val);


    // Outgoing dependencies for each task


    // Map tasks to rank
    auto task_2_rank = [&](int k) {
        return k / n_tasks_per_rank;
    };

    // Initialize the communicator structure
    Communicator comm(verb);

    // Initialize the runtime structures
    Threadpool tp(n_threads, &comm, verb, "WkTuto_" + to_string(rank) + "_");
    Taskflow<int> potrf(&tp, verb);
    Taskflow<int2> trsm(&tp, verb);
    Taskflow<int3> gemm(&tp, verb);

    // Create active message


    // Define the task flow
    potrf.set_task([&](int k) {
          printf("Potrf %d is now running on rank %d\n", k, comm_rank());
      })
        .set_fulfill([&](int k) {
            for (int p = k+1; p<nb; p++) // Looping through all outgoing dependency edges
            {
                int dest = task_2_rank(p); // defined above

                potrf.fulfill_promise(p, 5.0);
                printf("Potrf %d fulfilling local task %d on rank %d\n", k, p, comm_rank());

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
        printf("Trsm (%d, %d) is now running on rank %d\n", k, i, comm_rank());
      })
        .set_fulfill([&](int2 ki) {
            int k=ki[0];
            int i=ki[1];
            for (int j=k+1; j<nb;j++) // Looping through all outgoing dependency edges
            {

                if (i<j) {
                    gemm.fulfill_promise({k,i,j}, 5.0);
                    printf("Trsm (%d, %d) fulfilling local Gemm (%d, %d, %d) on rank %d\n", k, i, k, i, j, comm_rank());
                }
                else {
                    gemm.fulfill_promise({k,j,i}, 5.0);
                    printf("Trsm (%d, %d) fulfilling local Gemm (%d, %d, %d) on rank %d\n", k, i, k, j, i, comm_rank());
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

            return (k % n_threads);
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
            printf("Gemm (%d, %d, %d) is now running on rank %d\n", k, i, j, comm_rank());
      })
        .set_fulfill([&](int3 kij) {
            int k=kij[0];
            int i=kij[1];
            int j=kij[2];
            if (k<i-1) {
                gemm.fulfill_promise({k+1, i, j}, 5.0);
            }
            else {
                if (i==j) {
                    potrf.fulfill_promise(i, 5.0);
                    printf("Gemm (%d, %d, %d) fulfilling Potrf %d on rank %d\n", k, i, j, i, comm_rank());
                }
                else {
                    trsm.fulfill_promise({i,j}, 5.0);
                    printf("Gemm (%d, %d, %d) fulfilling Trsm (%d, %d) on rank %d\n", k, i, j, i, j, comm_rank());
                }
            }
            

        })
        .set_indegree([&](int3 kij) {
            int k=kij[0];
            if (k==0) {
                return 2;
            }
            else {
                return 3;
            }
        })
        .set_mapping([&](int3 kij) {
            int k=kij[0];
            int i=kij[1];
            int j=kij[2];

            return (k % n_threads);
        })
        .set_binding([&](int3 kij) {
            return false;

        })
        .set_name([&](int3 kij) { // This is just for debugging and profiling
            int k=kij[0];
            int i=kij[1];
            int j=kij[2];
            return "GEMM" + to_string(k) + "_" + to_string(i)+"_"+to_string(j)+"_"+to_string(comm_rank());
        });


    

    // Seed initial tasks
    potrf.fulfill_promise(0, 5.0);

    // Other ranks do nothing
    // Run until completion
    tp.join();
}

int main(int argc, char **argv)
{
    int req = MPI_THREAD_FUNNELED;
    int prov = -1;

    MPI_Init_thread(NULL, NULL, req, &prov);

    assert(prov == req);



    int n_threads = 2;
    int verb = 0; // Can be changed to vary the verbosity of the messages

    if (argc >= 2)
    {
        n_threads = atoi(argv[1]);
    }

    if (argc >= 3)
    {
        verb = atoi(argv[2]);
    }

    tuto_1(n_threads, verb);

    MPI_Finalize();
}
