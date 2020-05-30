#ifndef TTOR_SHARED

#include <vector>
#include <string>
#include <numeric>
#include <algorithm>

#include "threadpool_dist.hpp"

namespace ttor {

// Update counts on master
// We use step, provided by the worker, to update msg_queued and msg_processed with the latest information
void Threadpool_dist::set_msg_counts_master(int from, int msg_queued, int msg_processed) {
    if (verb > 1) {
        printf("[%s] <- %d, Message counts (%d %d)\n", name.c_str(), from, msg_queued, msg_processed);
    }
    assert(my_rank == 0);
    assert(from >= 0 && from < comm_size());
    assert(msgs_queued[from] >= -1);
    assert(msgs_processed[from] >= -1);
    assert(msg_queued >= 0);
    assert(msg_processed >= 0);
    msgs_queued[from] = std::max(msgs_queued[from], msg_queued);
    msgs_processed[from] = std::max(msgs_processed[from], msg_processed);
}

// Ask confirmation on worker
// If step is the latest information send, and if we're still idle and there were no new messages in between, reply with the tag
void Threadpool_dist::ask_confirmation(int msg_queued, int msg_processed, int tag) {
    assert(my_rank != 0);
    assert(msg_queued >= 0);
    assert(msg_processed >= 0);
    if (verb > 1) {
        printf("[%s] <- %d, Confirmation request tag %d (%d %d)\n", name.c_str(), 0, tag, msg_queued, msg_processed);
    }
    if(tag > last_rcvd_conf_tag) {
        last_rcvd_conf_tag = tag;
        last_rcvd_conf_nqueued = msg_queued;
        last_rcvd_conf_nprocessed = msg_processed;
    }
}

// Update tags on master with the latest confirmation tag
void Threadpool_dist::confirm(int from, int tag) {
    if (verb > 1) {
        printf("[%s] <- %d, Confirmation OK tag %d\n", name.c_str(), from, tag);
    }
    assert(my_rank == 0);
    assert(from >= 0 && from < comm_size());
    tags[from] = std::max(tags[from], tag);
}

// Shut down the TF
void Threadpool_dist::shutdown_tf() {
    if (verb > 0) {
        printf("[%s] Shutting down tf\n", name.c_str());
    }
    assert(tasks_in_flight.load() == 0);
    done.store(true);
}

// Everything is done in join
void Threadpool_dist::test_completion() {
    // Nothing
}

Threadpool_dist::Threadpool_dist(int n_threads, Communicator *comm_, int verb_, std::string basename_, bool start_immediately)
    : Threadpool_shared(n_threads, verb_, basename_, false),
        my_rank(comm_rank()),
        msgs_queued(comm_size(), -1),       // -1 means no count received yet             [rank 0 only]
        msgs_processed(comm_size(), -1),    // -1 means no count received yet             [rank 0 only]
        tags(comm_size(), -1),              // -1 means no confirmation tag received yet  [rank 0 only]
        intern_queued(0),
        intern_processed(0),
        last_sent_nqueued(-1),
        last_sent_nprocessed(-1),
        last_sent_conf_tag(-1),
        last_rcvd_conf_tag(-1),
        last_rcvd_conf_nqueued(-1),
        last_rcvd_conf_nprocessed(-1),
        last_sum(-1),                      // -1 means no sum computed yet [rank 0 only]
        comm(comm_),
        name(basename + "MPI_MASTER"),
        am_set_msg_counts_master(nullptr), // AM to send msg received and sent
        am_ask_confirmation(nullptr),
        am_send_confirmation(nullptr),
        am_shutdown_tf(nullptr), // AM to shut down the worker threads
        confirmation_tag(0)
{
    // Update message counts on master
    am_set_msg_counts_master = comm->make_active_msg(
        [&](int &from, int &msg_queued, int &msg_processed) {
            set_msg_counts_master(from, msg_queued, msg_processed);
            intern_processed++;
        });

    // Ask worker for confirmation on the latest count
    am_ask_confirmation = comm->make_active_msg(
        [&](int &msg_queued, int &msg_processed, int &tag) {
            ask_confirmation(msg_queued, msg_processed, tag);
            intern_processed++;
        });

    // Send confirmation to master
    am_send_confirmation = comm->make_active_msg(
        [&](int& from, int &tag) {
            confirm(from, tag);
            intern_processed++;
        });

    // Shutdown worker or master
    am_shutdown_tf = comm->make_active_msg(
        [&]() {
            shutdown_tf();
            intern_processed++;
        });

    // Now it is safe to call start()
    if (start_immediately)
        start();
}

void Threadpool_dist::join()
{
    assert(tasks_in_flight.load() > 0);
    --tasks_in_flight;
    // We can safely decrement tasks_in_flight.
    // All tasks have been seeded by the main thread.

    // We first exhaust all the TFs
    while (! is_done())
    {
        // We do as much progress as possible, on both the user and internal comm's
        do {
            comm->progress();
        } while (! comm->is_done());
        // If there is nothing to do, we check for completion
        // We may or not be done by now
        if ( (! is_done()) && tasks_in_flight.load() == 0 ) {
            test_completion_join();
        }
    }
    assert(tasks_in_flight.load() == 0);
    assert(is_done());

    while(! comm->is_done()) {
        comm->progress();
    }

    assert(comm->is_done());

    // All threads join
    all_threads_join();
}

// Return the number of internal queued rpcs
int Threadpool_dist::get_intern_n_msg_queued() {
    int nq = comm->get_n_msg_queued() - intern_queued;
    assert(nq >= 0);
    return nq;
}

// Return the number of internal processed lpcs
int Threadpool_dist::get_intern_n_msg_processed() {
    int np = comm->get_n_msg_processed() - intern_processed;
    assert(np >= 0);
    return np;
}

// Only MPI Master thread runs this function, on all ranks
// When done, this function set done to true and is_done() now returns true
void Threadpool_dist::test_completion_join()
{

    /**
     * Strategy
     * 
     * We send 4 kinds of messages
     * - Rank !=0 -> Rank 0: latest user rpcs/lpcs counts with a 'count' AM
     * - Rank   0 -> Rank != 0: when all rpcs/lpcs counts match, sends a 'request' AM
     *      We associate 'request' with a unique tag
     * - Rank !=0 -> Rank 0: reply to the latest 'request' if the counts still match with a 'confirmation' AM
     *      The replies use the latest received tag
     * - Rank   0 -> Rank != 0: when all ranks reply to the latest sent confirmation request with their confirmation, we send a 'shutdown' AM
     * 
     * Rank 0 can do two things:
     * - Check the latest 'confirmation'. If we got a positive reply from all ranks for the _latest_ request, we send a 'shutdown'
     * - Otherwise, check the rpcs/lpcs 'count's. If they all match, we send a 'request' to all other ranks
     * 
     * Rank != 0 can do two things:
     * - If we have a new rpcs/lpcs count, send the 'count'
     * - If we have unanswered 'request', we look at the latest. If the count haven't changed in the meantime, send a 'confirmation' back to rank 0
     * 
     * Observations:
     * - The internal comms send a finite number of messages, assuming the TF sends a finite number of messages.
     * 
     * - It will terminate. When the TF is empty, all count match, and the confirmation request will be positively replied to by all ranks
     * 
     * - When the final 'confirmation' is sent from rank != 0, no more message will be sent from rank 0
     *      - Previous 'counts' or 'confirmations' arrive on rank 0 before the ultimate confirmation
     *      - Only that final confirmation request can trigger shutdown
     *      - Hence, rank 0 can terminate with a progress-terminate loop
     * - When that shutdown reaches rank != 0, all previous message already reached rank != 0
     *      - Hence, rank != 0 can terminate with a progress-terminate loop
     * 
     * Those four facts show that
     * (1) Only a finite number of messages are created (prevents flooding or fairness issue)
     * (2) We eventually terminate
     * (3) When we terminate, there are no pending MPI message
     **/

    // No tasks are running in the threadpool so noone can queue rpcs
    // MPI thread is the one running comm->progress(), so it can check is_done() properly, no race conditions here
    int my_rank = comm_rank();
    assert(! is_done());
    assert(tasks_in_flight.load() == 0);
    if(my_rank == 0) {
        // STEP A: check the previously received confirmation tags
        // If we got an anser from all ranks for the latest confirmation_tag, we terminate
        const bool all_tags_ok = std::all_of(tags.begin(), tags.end(), [&](int t){return t == confirmation_tag;});
        if(all_tags_ok) {
            if (verb > 1) {
                printf("[%s] all tags OK\n", name.c_str());
            }
            for(int r = 1; r < comm_size(); r++) {
                intern_queued++;
                am_shutdown_tf->send(r);
            }
            shutdown_tf();
        }
        // STEP B: check the nqueued and nprocessed, send confirmations
        // If they match, send request to workers
        else {
            int nq = get_intern_n_msg_queued();
            int np = get_intern_n_msg_processed();
            set_msg_counts_master(0, nq, np);
            const int queued_sum    = std::accumulate(msgs_queued.begin(),    msgs_queued.end(), 0, std::plus<int>());
            const int processed_sum = std::accumulate(msgs_processed.begin(), msgs_processed.end(), 0, std::plus<int>());
            const bool all_updated  = std::all_of(msgs_queued.begin(), msgs_queued.end(), [](int i){return i >= 0;});
            // If they match and we have a new count, ask worker for confirmation (== synchronization)
            if(all_updated && processed_sum == queued_sum && last_sum != processed_sum) {
                confirmation_tag++;
                if (verb > 0) {
                    printf("[%s] processed_sum == queued_sum == %d, asking confirmation %d\n", name.c_str(), processed_sum, confirmation_tag);
                }
                for(int r = 1; r < comm_size(); r++) {
                    intern_queued++;
                    am_ask_confirmation->send(r, msgs_queued[r], msgs_processed[r], confirmation_tag);
                }
                tags[0] = confirmation_tag;
                last_sum = processed_sum;
            }
        }
    } else {
        // STEP A: We send to 0 our updated counts, if they have changed
        {
            int nq = get_intern_n_msg_queued();
            int np = get_intern_n_msg_processed();
            bool new_values = (nq != last_sent_nqueued || np != last_sent_nprocessed);
            if(new_values) {
                intern_queued++;
                am_set_msg_counts_master->send(0, my_rank, nq, np);
                last_sent_nqueued = nq;
                last_sent_nprocessed = np;
                if (verb > 1) {
                    printf("[%d] -> 0 tif %d done %d sending %d %d\n", comm_rank(), tasks_in_flight.load(), (int)comm->is_done(), nq, np);
                }
            }
        }
        // STEP B: We reply to the latest confirmation request
        {
            assert(last_sent_conf_tag <= last_rcvd_conf_tag);
            if(last_sent_conf_tag < last_rcvd_conf_tag) {
                int nq = get_intern_n_msg_queued();
                int np = get_intern_n_msg_processed();
                if(nq == last_rcvd_conf_nqueued && np == last_rcvd_conf_nprocessed) {
                    if (verb > 1) {
                        printf("[%s] -> 0 Confirmation YES tag %d (%d %d)\n", name.c_str(), last_rcvd_conf_tag, nq, np);
                    }
                    int from = comm_rank();
                    intern_queued++;
                    am_send_confirmation->send(0, from, last_rcvd_conf_tag);
                    last_sent_conf_tag = last_rcvd_conf_tag;
                }
            }
        }
    }
}

} // namespace ttor {

#endif