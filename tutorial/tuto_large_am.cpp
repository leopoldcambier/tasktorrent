#include "tasktorrent/tasktorrent.hpp"

using namespace std;
using namespace ttor;

/**
 * This example show a simple code using the large active message functionality
 * We have 1 task per rank, where every task sends a large message on the next rank
 * and triggers the task there
 */
void tuto(int verb)
{
    const int n_threads = 1;
    const int rank = comms_world_rank();
    const int n_ranks = comms_world_size();

    if (n_ranks < 2)
    {
        printf("You need to run this code with at least 2 processors\n");
        exit(0);
    }

    printf("Rank %d hello\n", rank);

    // Prepare data to send and to receive
    // Data to send is filled with data
    // Data to receive is left unallocated
    const int ntasks = n_ranks;
    const int N = 1000;
    const vector<int> tosend(N, rank);
    vector<int> torecv;

    // Initialize the communicator structure
    auto comm = make_communicator_world(verb);

    // Initialize the runtime structures
    Threadpool tp(n_threads, comm.get(), verb, "WkTuto_" + to_string(rank) + "_");
    Taskflow<int> tf(&tp, verb);

    // We map 1 task per rank
    // Each task simply consists of sending the data in `tosend` to the next rank
    // where it is received in `torecv`

    // Create large active message
    // `from` indicates from who is the message coming
    // `k` is the task (== rank in this example)
    auto am = comm->make_large_active_msg(
        // This function is ran on the receiver when the buffer has arrived
        // This is typically used to trigger tasks using that data
        [&](const int& from, const int& k) {
            printf("Data from %d for task %d received on rank %d\n", from, k, rank);
            tf.fulfill_promise(k);
        },
        // This function is ran on the receiver to get the pointer to the buffer
        // in which to store the data
        // It should return a pointer to an allocated buffer large enough to hold the data
        [&](const int& from, const int& k) {
            printf("Data from %d for task %d starting to be received on rank %d\n", from, k, rank);
            torecv.resize(N);
            return torecv.data();
        },
        // This function is ran on the sender when the buffer can safely be reused
        // This would for instance free the buffer, if needed
        [&](const int& from, const int& k) {
            printf("Data from %d for task %d sent from %d\n", from, k, rank);
        });

    // Define the task flow
    tf.set_task([&](int k) {
            // At this point, since this task if fulfill by the AM, the data has arrived
            // Task 0 does not expect data
            if(k > 0) assert(torecv[0] == k-1);
            printf("Task %d is now running on rank %d\n", k, rank);
        })
        .set_fulfill([&](int k) {
            // Send data to the next rank
            if(k < ntasks-1) {
                // The first argument of send_large is the destination
                // Followed by a view to a buffer
                // Followed by the usual arguments
                am->send_large(rank+1, make_view(tosend.data(), tosend.size()), rank, rank+1);
            }
        })
        .set_indegree([&](int k) {
            return 1;
        })
        .set_mapping([&](int k) {
            return 0;
        })
        .set_name([&](int k) { // This is just for debugging and profiling
            return "tutoTask_" + to_string(k) + "_" + to_string(rank);
        });

    // Seed initial tasks
    // The first task is task 0 on rank 0
    if (rank == 0)
    {
        tf.fulfill_promise(0); // Task 0 starts on rank 0
    }

    // Run until completion
    tp.join();
}

int main(int argc, char **argv)
{
    comms_init();

    int verb = 0; // Can be changed to vary the verbosity of the messages

    if (argc >= 2)
    {
        verb = atoi(argv[1]);
    }

    tuto(verb);

    comms_finalize();
}
