// make tests_completion && mpirun -oversubscribe -n 4 ./tests_completion 4 1 --gtest_repeat=16 --gtest_break_on_failure

#include "communications.hpp"
#include "runtime.hpp"
#include "util.hpp"

#include <iostream>
#include <map>

#include <mpi.h>

#include <gtest/gtest.h>

using namespace std;
using namespace ttor;

typedef Taskflow<int> tflow_t;

int n_threads = 2;
int verb = 0; // Can be changed to vary the verbosity of the messages

TEST(completion, mini)
{
    const int rank = comm_rank();
    const int n_ranks = comm_size();

    printf("Rank %d hello from %s\n", rank, processor_name().c_str());

    // Number of incoming dependencies for each task
    map<int, int> indegree;

    // Initialize the communicator structure
    Communicator comm(verb);

    // Initialize the runtime structures
    Threadpool tp(n_threads, &comm, verb, "Test1_" + to_string(rank) + "_");
    tflow_t tf(&tp, verb);

    atomic<int> test_var(0);

    // Define the tasks
    tf.set_mapping([&](int k) {
          return (k % n_threads);
      })
        .set_indegree([&](int k) {
            return 1;
        })
        .set_task([&](int k) {
            if (verb > 0)
                printf("DAG Test1 index %d running on rank %d\n", k, comm_rank());
            ++test_var;
        })
        .set_name([&](int k) { // This is just for debugging and profiling
            return "DAG_Test1_idx_" + to_string(k) + "_rank_" + to_string(rank);
        });

    // Seed initial tasks
    for (int i = 0; i < n_threads; ++i)
        tf.fulfill_promise(i); // Task 0 starts on rank 0

    // Run until completion
    tp.join();
    // Done!

    ASSERT_EQ(test_var.load(), n_threads);

    MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char **argv)
{
    int req = MPI_THREAD_FUNNELED;
    int prov = -1;

    MPI_Init_thread(NULL, NULL, req, &prov);

    assert(prov == req);

    if (argc >= 2)
    {
        n_threads = atoi(argv[1]);
    }

    if (argc >= 3)
    {
        verb = atoi(argv[2]);
    }

    ::testing::InitGoogleTest(&argc, argv);

    const int return_flag = RUN_ALL_TESTS();

    MPI_Finalize();
    return return_flag;
}
