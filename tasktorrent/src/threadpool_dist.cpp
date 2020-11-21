#if defined(TTOR_MPI) || (TTOR_UPCXX)

#include <vector>
#include <string>
#include <numeric>
#include <algorithm>

#include "util.hpp"
#include "threadpool_dist.hpp"

namespace ttor {

// Update counts on master
template<typename C_t>
void Threadpool_dist<C_t>::set_msg_counts_master(int from, llint msg_queued, llint msg_processed) {
    if (verb > 1) {
        printf("[%s] <- %d, Message counts (%lld %lld)\n", name.c_str(), from, msg_queued, msg_processed);
    }
    assert(my_rank == 0);
    assert(from >= 0 && from < num_ranks);
    assert(msgs_queued[from] >= -1);
    assert(msgs_processed[from] >= -1);
    assert(msg_queued >= 0);
    assert(msg_processed >= 0);
    msgs_queued[from] = std::max(msgs_queued[from], msg_queued);
    msgs_processed[from] = std::max(msgs_processed[from], msg_processed);
}

// Ask confirmation on worker
template<typename C_t>
void Threadpool_dist<C_t>::ask_confirmation(llint msg_queued, llint msg_processed, llint tag) {
    assert(my_rank != 0);
    assert(msg_queued >= 0);
    assert(msg_processed >= 0);
    if (verb > 1) {
        printf("[%s] <- %d, Confirmation request tag %lld (%lld %lld)\n", name.c_str(), 0, tag, msg_queued, msg_processed);
    }
    if(tag > last_rcvd_conf_tag) {
        last_rcvd_conf_tag = tag;
        last_rcvd_conf_nqueued = msg_queued;
        last_rcvd_conf_nprocessed = msg_processed;
    }
}

// Update tags on master with the latest confirmation tag
template<typename C_t>
void Threadpool_dist<C_t>::confirm(int from, llint tag, llint intern_queued_from_nonmaster) {
    if (verb > 1) {
        printf("[%s] <- %d, Confirmation OK tag %lld, intern queued %lld\n", name.c_str(), from, tag, intern_queued_from_nonmaster);
    }
    assert(intern_queued_from_nonmaster >= 0);
    assert(my_rank == 0);
    assert(from >= 0 && from < num_ranks);
    intern_msgs_queued_from_nonmaster[from] = std::max(intern_msgs_queued_from_nonmaster[from], intern_queued_from_nonmaster);
    tags[from] = std::max(tags[from], tag);
}

// Shut down the TF
template<typename C_t>
void Threadpool_dist<C_t>::shutdown_tf(llint intern_queued_from_master) {
    if (verb > 0) {
        printf("[%s] Shutting down tf, intern queued %lld\n", name.c_str(), intern_queued_from_master);
    }
    assert(intern_queued_from_master >= 0);
    intern_msgs_queued_from_master = intern_queued_from_master;
    assert(tasks_in_flight.load() == 0);
    done.store(true);
}

// Everything is done in join
template<typename C_t>
void Threadpool_dist<C_t>::test_completion() {
    // Nothing
}

template<typename C_t>
Threadpool_dist<C_t>::Threadpool_dist(int n_threads, C_t *comm_, int verb_, std::string basename_, bool start_immediately)
    : Threadpool_shared(n_threads, verb_, basename_, false),
        my_rank(comm_->comm_rank()),
        num_ranks(comm_->comm_size()),
        msgs_queued(num_ranks, -1),       // -1 means no count received yet             [rank 0 only]
        msgs_processed(num_ranks, -1),    // -1 means no count received yet             [rank 0 only]
        tags(num_ranks, -1),              // -1 means no confirmation tag received yet  [rank 0 only]
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
        confirmation_tag(0),
        intern_msgs_queued_from_nonmaster(num_ranks, -1), // -1 means no count received yet
        intern_msgs_queued_to_nonmaster(num_ranks, 0),
        intern_msgs_queued_from_master(-1)                // -1 means no count received yet
{
    // Update message counts on master
    am_set_msg_counts_master = comm->make_active_msg(
        [&](int from, llint msg_queued, llint msg_processed) {
            set_msg_counts_master(from, msg_queued, msg_processed);
            intern_processed++;
        });

    // Ask worker for confirmation on the latest count
    am_ask_confirmation = comm->make_active_msg(
        [&](llint msg_queued, llint msg_processed, llint tag) {
            ask_confirmation(msg_queued, msg_processed, tag);
            intern_processed++;
        });

    // Send confirmation to master
    am_send_confirmation = comm->make_active_msg(
        [&](int from, llint tag, llint intern_queued_from_nonmaster) {
            confirm(from, tag, intern_queued_from_nonmaster);
            intern_processed++;
        });

    // Shutdown worker or master
    am_shutdown_tf = comm->make_active_msg(
        [&](llint intern_queued_from_master) {
            shutdown_tf(intern_queued_from_master);
            intern_processed++;
        });

    // Now it is safe to call start()
    if (start_immediately)
        start();
}

template<typename C_t>
llint Threadpool_dist<C_t>::internal_finish_counts() 
{
    // We verify that counts have been updated, which they should have been
    if(my_rank == 0) {
        assert(std::all_of(intern_msgs_queued_from_nonmaster.begin(), intern_msgs_queued_from_nonmaster.end(), [&](llint t){return t >= 0;}));
        return std::accumulate(intern_msgs_queued_from_nonmaster.begin(), 
                               intern_msgs_queued_from_nonmaster.end(), 
                               0, std::plus<llint>());
    } else {
        assert(intern_msgs_queued_from_master >= 0);
        return intern_msgs_queued_from_master;
    }
}

template<typename C_t>
void Threadpool_dist<C_t>::join()
{
    assert(tasks_in_flight.load() > 0);
    --tasks_in_flight;
    // We can safely decrement tasks_in_flight.
    // All tasks have been seeded by the main thread.

    // We first exhaust all the user TFs
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
    assert(intern_processed <= internal_finish_counts());

    // We then continue until all internal communications have (1) been sent (2) have been processed
    // There is no risk of bypassing this since, by now, user comms have completed so
    // (1) shutdown has been received (2) shutdown can only be send after matching counts
    // So we are guaranteed that counts have been updated
    while( (!comm->is_done()) || (intern_processed != internal_finish_counts())) {
        comm->progress();
    }

    assert(intern_processed == internal_finish_counts());
    assert(comm->is_done());

    // All threads join
    all_threads_join();
}

// Return the number of user queued rpcs
template<typename C_t>
llint Threadpool_dist<C_t>::get_user_n_msg_queued() {
    llint nq = comm->get_n_msg_queued() - intern_queued;
    assert(nq >= 0);
    return nq;
}

// Return the number of user processed lpcs
template<typename C_t>
llint Threadpool_dist<C_t>::get_user_n_msg_processed() {
    llint np = comm->get_n_msg_processed() - intern_processed;
    assert(np >= 0);
    return np;
}

// Only MPI Master thread runs this function, on all ranks
// When done, this function set done to true and is_done() now returns true
template<typename C_t>
void Threadpool_dist<C_t>::test_completion_join()
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
     * - In parallel of the above, we count the number of messages from rank 0 to rank != 0
     *     - When the shutdown message is sent, we update the number of queued messages from rank 0 to ranks != 0
     *     - During every confirmation, we update the number of queued messages from rank != 0 to rank 0
     *   This ensures that whenever we are done with user AMs (all completed) and are done queuing internal AM (but not all     
     *   completed), those counts are fully updated
     *   When can then safely call progress until (1) queues are empty and (2) the counts of expected internal AM match
     * 
     *   The advantage is that this does not rely on any particular property of the communication layer such as ordering
     * 
     * Those four facts show that
     * (1) Only a finite number of messages are created (prevents flooding or fairness issue)
     * (2) We eventually terminate
     * (3) When we terminate, there are no pending MPI message
     **/

    // No tasks are running in the threadpool so noone can queue rpcs
    // Main thread is the one running comm->progress(), so it can check is_done() properly, no race conditions here
    assert(! is_done());
    assert(tasks_in_flight.load() == 0);
    if(my_rank == 0) {
        // STEP A: check the previously received confirmation tags
        // If we got an anser from all ranks for the latest confirmation_tag, we terminate
        // This would then be the final queued message from rank 0 to rank != 0
        const bool all_tags_ok = std::all_of(tags.begin(), tags.end(), [&](llint t){return t == confirmation_tag;});
        if(all_tags_ok) {
            if (verb > 1) {
                printf("[%s] all tags OK\n", name.c_str());
            }
            for(int r = 1; r < num_ranks; r++) {
                intern_queued++;
                intern_msgs_queued_to_nonmaster[r]++;
                am_shutdown_tf->send(r, intern_msgs_queued_to_nonmaster[r]);
            }
            shutdown_tf(0);
        }
        // STEP B: check the nqueued and nprocessed, send confirmations
        // If they match, send request to workers
        else {
            llint nq = get_user_n_msg_queued();
            llint np = get_user_n_msg_processed();
            set_msg_counts_master(0, nq, np);
            const llint queued_sum    = std::accumulate(msgs_queued.begin(),    msgs_queued.end(), 0, std::plus<llint>());
            const llint processed_sum = std::accumulate(msgs_processed.begin(), msgs_processed.end(), 0, std::plus<llint>());
            const bool all_updated  = std::all_of(msgs_queued.begin(), msgs_queued.end(), [](llint i){return i >= 0;});
            // If they match and we have a new count, ask worker for confirmation (== synchronization)
            if(all_updated && processed_sum == queued_sum && last_sum != processed_sum) {
                confirmation_tag++;
                if (verb > 0) {
                    printf("[%s] processed_sum == queued_sum == %lld, asking confirmation %lld\n", name.c_str(), processed_sum, confirmation_tag);
                }
                for(int r = 1; r < num_ranks; r++) {
                    intern_queued++;
                    intern_msgs_queued_to_nonmaster[r]++;
                    am_ask_confirmation->send(r, msgs_queued[r], msgs_processed[r], confirmation_tag);
                }
                tags[0] = confirmation_tag;
                intern_msgs_queued_from_nonmaster[0] = 0;
                last_sum = processed_sum;
            }
        }
    } else {
        // STEP A: We send to 0 our updated counts, if they have changed
        {
            llint nq = get_user_n_msg_queued();
            llint np = get_user_n_msg_processed();
            bool new_values = (nq != last_sent_nqueued || np != last_sent_nprocessed);
            if(new_values) {
                intern_queued++;
                am_set_msg_counts_master->send(0, my_rank, nq, np);
                last_sent_nqueued = nq;
                last_sent_nprocessed = np;
                if (verb > 1) {
                    printf("[%d] -> 0 tif %lld done %d sending %lld %lld\n", my_rank, tasks_in_flight.load(), (int)comm->is_done(), nq, np);
                }
            }
        }
        // STEP B: We reply to the latest confirmation request
        // Whenever completion is reached, this will be the final message
        {
            assert(last_sent_conf_tag <= last_rcvd_conf_tag);
            if(last_sent_conf_tag < last_rcvd_conf_tag) {
                llint nq = get_user_n_msg_queued();
                llint np = get_user_n_msg_processed();
                if(nq == last_rcvd_conf_nqueued && np == last_rcvd_conf_nprocessed) {
                    if (verb > 1) {
                        printf("[%s] -> 0 Confirmation YES tag %lld (%lld %lld)\n", name.c_str(), last_rcvd_conf_tag, nq, np);
                    }
                    intern_queued++;
                    am_send_confirmation->send(0, my_rank, last_rcvd_conf_tag, intern_queued);
                    last_sent_conf_tag = last_rcvd_conf_tag;
                }
            }
        }
    }
}

#if defined(TTOR_MPI)

template class Threadpool_dist<Communicator_MPI>;

#elif defined(TTOR_UPCXX)

template class Threadpool_dist<Communicator_UPCXX>;

#endif

} // namespace ttor {

#endif