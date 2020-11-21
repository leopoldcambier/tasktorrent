#include <iostream>
#include <map>
#include <algorithm>
#include <cmath>

#include <gtest/gtest.h>

#include "tasktorrent/tasktorrent.hpp"

using namespace std;
using namespace ttor;

typedef Taskflow<int> tflow_t;

int N_THREADS = 2;
int VERB = 0; // Can be changed to vary the verbosity of the messages

TEST(Completion, Mini)
{
    const int rank = comms_world_rank();

    // Initialize the communicator structure
    auto comm = make_communicator_world(VERB);

    // Initialize the runtime structures
    Threadpool tp(N_THREADS, comm.get(), VERB, "mini_" + to_string(rank) + "_");
    tflow_t tf(&tp, VERB);

    atomic<int> test_var(0);

    // Define the tasks
    tf.set_mapping([&](int k) {
            return (k % N_THREADS);
        })
        .set_indegree([&](int) {
            return 1;
        })
        .set_task([&](int k) {
            if (VERB > 0)
                printf("mini %d running on rank %d\n", k, comms_world_rank());
            ++test_var;
        })
        .set_name([&](int k) { // This is just for debugging and profiling
            return "mini_" + to_string(k) + "_rank_" + to_string(rank);
        });

    // Seed initial tasks
    for (int i = 0; i < N_THREADS; ++i)
        tf.fulfill_promise(i); // Task 0 starts on rank 0

    // Run until completion
    tp.join();
    // Done!

    ASSERT_EQ(test_var.load(), N_THREADS);

    comms_world_barrier();
}

TEST(Completion, Line)
{
    const int rank = comms_world_rank();
    const int ntasks = comms_world_size() * N_THREADS;

    auto task_2_rank = [&](int k){
        return k / N_THREADS;
    };

    // Initialize the communicator structure
    auto comm = make_communicator_world(VERB);

    // Initialize the runtime structures
    Threadpool tp(N_THREADS, comm.get(), VERB, "line_" + to_string(rank) + "_");
    tflow_t tf(&tp, VERB);
    auto am = comm->make_active_msg(
        [&](const int &k) {
            ASSERT_EQ(task_2_rank(k), rank);
            tf.fulfill_promise(k);
        });

    atomic<int> tasks_done(0);

    // Define the tasks
    tf.set_mapping([&](int k) {
            return (k % N_THREADS);
        })
        .set_indegree([&](int) {
            return 1;
        })
        .set_task([&](int k) {
            if (VERB > 0)
                printf("line TF %d running on rank %d\n", k, comms_world_rank());
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

    ASSERT_EQ(tasks_done.load(), N_THREADS);

    comms_world_barrier();
}

TEST(Completion, TrivialUnequal)
{
    const int rank = comms_world_rank();
    const int ntasks = N_THREADS * (int)std::min(64000.0, std::pow(16, std::min(8,rank)));

    // Initialize the communicator structure
    auto comm = make_communicator_world(VERB);

    // Initialize the runtime structures
    Threadpool tp(N_THREADS, comm.get(), VERB, "line_" + to_string(rank) + "_");
    tflow_t tf(&tp, VERB);

    atomic<int> tasks_done(0);

    // Define the tasks
    tf.set_mapping([&](int k) {
            return (k % N_THREADS);
        })
        .set_indegree([&](int) {
            return 1;
        })
        .set_task([&](int k) {
            int next_k = k+1;
            if(next_k < ntasks) {
                tf.fulfill_promise(next_k);
            }
            ++tasks_done;
        });

    // Seed initial task
    tf.fulfill_promise(0);

    // Run until completion
    tp.join();

    ASSERT_EQ(tasks_done.load(), ntasks);

    comms_world_barrier();
}

int main(int argc, char **argv)
{
    comms_init();
    ::testing::InitGoogleTest(&argc, argv);

    if (argc >= 2)
    {
        N_THREADS = atoi(argv[1]);
    }

    if (argc >= 3)
    {
        VERB = atoi(argv[2]);
    }

    const int return_flag = RUN_ALL_TESTS();
    comms_finalize();
    return return_flag;
}
