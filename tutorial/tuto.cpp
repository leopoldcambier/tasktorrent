#include "tasktorrent/tasktorrent.hpp"

#include <iostream>
#include <map>
#include <functional>

using namespace std;
using namespace ttor;

void tuto(int n_threads, int verb)
{
    const int rank = comms_world_rank();
    const int n_ranks = comms_world_size();

    if (n_ranks < 2)
    {
        printf("You need to run this code with at least 2 processors\n");
        exit(0);
    }

    printf("Rank %d hello on %s\n", rank, comms_hostname().c_str());

    // Number of tasks
    int n_tasks_per_rank = 2;

    // Outgoing dependencies for each task
    map<int, vector<int>> out_deps;
    out_deps[0] = {1, 3}; // Task 0 fulfills task 1 and 3
    out_deps[2] = {1, 3}; // Task 2 fulfills task 1 and 3

    // Number of incoming dependencies for each task
    map<int, int> indegree;
    indegree[0] = 1;
    indegree[1] = 2;
    indegree[2] = 1;
    indegree[3] = 2;

    // Map tasks to rank
    auto task_2_rank = [&](int k) {
        return k / n_tasks_per_rank;
    };

    // Initialize the communicator structure
    auto comm = make_communicator_world();

    // Initialize the runtime structures
    Threadpool tp(n_threads, comm.get(), verb, "WkTuto_" + to_string(rank) + "_");
    Taskflow<int> tf(&tp, verb);

    // Create active message
    auto am = comm->make_active_msg(
        [&](const int &k, const int &k_) {
            /* The data k and k_ are received over the network using MPI */
            printf("Task %d fulfilling %d (remote)\n", k, k_);
            tf.fulfill_promise(k_);
        });

    // Define the task flow
    tf.set_task([&](int k) {
          printf("Task %d is now running on rank %d\n", k, rank);
        })
        .set_fulfill([&](int k) {
            for (int k_ : out_deps[k]) // Looping through all outgoing dependency edges
            {
                int dest = task_2_rank(k_); // defined above
                if (dest == rank)
                {
                    tf.fulfill_promise(k_);
                    printf("Task %d fulfilling local task %d on rank %d\n", k, k_, rank);
                }
                else
                {
                    // Satisfy remote task
                    // Send k and k_ to rank dest using an MPI non-blocking send.
                    // The type of k and k_ must match the declaration of am above.
                    // am->send(dest, k, k_);
                    am->send(dest, k, k_);
                }
            }
        })
        .set_indegree([&](int k) {
            return indegree[k];
        })
        .set_mapping([&](int k) {
            /* This is the index of the thread that will get assigned to run this task.
             * Tasks can in general be stolen (i.e., migrate) by other threads, when idle.
             * The optional set_binding function below determines whether task k
             * is migratable or not.
             */
            return (k % n_threads);
        })
        .set_binding([&](int k) {
            return false;
            /* false == task can be migrated between worker threads [default value].
             * true == task is bound to the thread selected by set_mapping.
             * This function is optional. The library assumes false if this
             * function is not defined.
             */
        })
        .set_name([&](int k) { // This is just for debugging and profiling
            return "tutoTask_" + to_string(k) + "_" + to_string(rank);
        });

    // Seed initial tasks
    if (rank == task_2_rank(0))
    {
        tf.fulfill_promise(0); // Task 0 starts on rank 0
    }
    else if (rank == task_2_rank(2))
    {
        tf.fulfill_promise(2); // Task 2 starts on rank 1
    }

    // Other ranks do nothing
    // Run until completion
    tp.join();
}

int main(int argc, char **argv)
{
    comms_init();

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

    tuto(n_threads, verb);

    comms_finalize();
}
