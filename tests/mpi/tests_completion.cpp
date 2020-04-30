#include <iostream>
#include <map>
#include <gtest/gtest.h>
#include <mpi.h>

#include "tasktorrent/tasktorrent.hpp"

using namespace std;
using namespace ttor;

typedef Taskflow<int> tflow_t;

int n_threads = 2;
int verb = 0; // Can be changed to vary the verbosity of the messages

TEST(completion, mini)
{
    const int rank = comm_rank();

    // Initialize the communicator structure
    Communicator comm(verb);

    // Initialize the runtime structures
    Threadpool tp(n_threads, &comm, verb, "mini_" + to_string(rank) + "_");
    tflow_t tf(&tp, verb);

    atomic<int> test_var(0);

    // Define the tasks
    tf.set_mapping([&](int k) {
            return (k % n_threads);
        })
        .set_indegree([&](int) {
            return 1;
        })
        .set_task([&](int k) {
            if (verb > 0)
                printf("mini %d running on rank %d\n", k, comm_rank());
            ++test_var;
        })
        .set_name([&](int k) { // This is just for debugging and profiling
            return "mini_" + to_string(k) + "_rank_" + to_string(rank);
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

TEST(completion, line)
{
    const int rank = comm_rank();
    const int ntasks = comm_size() * n_threads;

    auto task_2_rank = [&](int k){
        return k / n_threads;
    };

    // Initialize the communicator structure
    Communicator comm(verb);

    // Initialize the runtime structures
    Threadpool tp(n_threads, &comm, verb, "line_" + to_string(rank) + "_");
    tflow_t tf(&tp, verb);
    auto am = comm.make_active_msg(
        [&](int &k) {
            ASSERT_EQ(task_2_rank(k), rank);
            tf.fulfill_promise(k);
        });

    atomic<int> tasks_done(0);

    // Define the tasks
    tf.set_mapping([&](int k) {
            return (k % n_threads);
        })
        .set_indegree([&](int) {
            return 1;
        })
        .set_task([&](int k) {
            if (verb > 0)
                printf("line TF %d running on rank %d\n", k, comm_rank());
            int next_k = k+1;
            if(next_k < ntasks) { // not the last task
                int dest = task_2_rank(next_k);
                if(dest != rank) {
                    am->send(dest, next_k);
                } else {
                    tf.fulfill_promise(next_k);
                }
            }
            ++tasks_done;
        })
        .set_name([&](int k) { // This is just for debugging and profiling
            return "line_" + to_string(k) + "_rank_" + to_string(rank);
        });

    // Seed initial task
    if(task_2_rank(0) == rank) {
        tf.fulfill_promise(0); // Task 0 starts
    }

    // Run until completion
    tp.join();

    ASSERT_EQ(tasks_done.load(), n_threads);

    MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char **argv)
{
    int req = MPI_THREAD_FUNNELED;
    int prov = -1;

    MPI_Init_thread(NULL, NULL, req, &prov);

    assert(prov == req);

    ::testing::InitGoogleTest(&argc, argv);

    if (argc >= 2)
    {
        n_threads = atoi(argv[1]);
    }

    if (argc >= 3)
    {
        verb = atoi(argv[2]);
    }

    const int return_flag = RUN_ALL_TESTS();

    MPI_Finalize();
    return return_flag;
}
