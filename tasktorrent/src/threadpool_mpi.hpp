#ifndef __TTOR_SRC_THREADPOOL_MPI_HPP__
#define __TTOR_SRC_THREADPOOL_MPI_HPP__

#ifndef TTOR_SHARED

#include <vector>
#include <string>

#include "util.hpp"
#include "tasks.hpp"
#include "threadpool_shared.hpp"
#include "communications.hpp"
#include "active_messages.hpp"

namespace ttor 
{

/**
 * \brief A threadpool associated to active messages and a communicator
 * 
 * \details This class behaves exactly like Threadpool_shared
 *          except that `join()` is overloaded to complete 
 *          when all threadpools on all ranks have completed
 *          and there are no in-flight active messages 
 *          (i.e. all queued rpcs have been processed)
 */
class Threadpool_mpi : public Threadpool_shared
{
    
private:

    const int my_rank;

    // Global count of all the messages sent and received
    // Used on rank 0
    std::vector<int>  msgs_queued;       // msgs_queued[from]: user's comm queued rpcs from rank from
    std::vector<int>  msgs_processed;    // msgs_processed[from]: user's comm processed lpcs from rank from
    std::vector<int>  tags;              // tags[from]: the greatest ever received confirmation tag

    // Count the number of queued and processed AM's used in the join()
    // Used on all ranks
    int intern_queued;     // the number of internal queued rpcs
    int intern_processed;  // the number of internal processed lpcs

    // The last information used/send
    // Used on all ranks except 0
    int last_sent_nqueued;    // the last sent value of user's queued rpcs
    int last_sent_nprocessed; // the last sent value of usue's processed lpcs

    // The last confirmaton request and confirmation information
    // Used on all ranks except 0
    int last_sent_conf_tag;
    int last_rcvd_conf_tag;
    int last_rcvd_conf_nqueued;
    int last_rcvd_conf_nprocessed;

    // Last sum, the last sum_r queued(r) == sum_r processed(r) value checked
    // Used on rank 0
    int last_sum;  
    
    // Used by user to communicate
    Communicator *comm;
    std::string name;

    // Active messages used to determine whether quiescence has been reached
    ActiveMsg<int, int, int>    *am_set_msg_counts_master;   // Send msg count to master
    ActiveMsg<int, int, int>    *am_ask_confirmation;        // Ask worker for confirmation
    ActiveMsg<int, int>         *am_send_confirmation;       // Send confirmation back to master
    ActiveMsg<>                 *am_shutdown_tf;             // Shutdown TF (last message from master to worker)

    // To detect completion
    // Used on rank 0
    int confirmation_tag;

    // Update counts on master
    // We use step, provided by the worker, to update msg_queued and msg_processed with the latest information
    void set_msg_counts_master(int from, int msg_queued, int msg_processed);

    // Ask confirmation on worker
    // If step is the latest information send, and if we're still idle and there were no new messages in between, reply with the tag
    void ask_confirmation(int msg_queued, int msg_processed, int tag);

    // Update tags on master with the latest confirmation tag
    void confirm(int from, int tag);

    // Shut down the TF
    void shutdown_tf();

    // Everything is done in join
    void test_completion() override;

    // Return the number of internal queued rpcs
    int get_intern_n_msg_queued();

    // Return the number of internal processed lpcs
    int get_intern_n_msg_processed();

    // Only MPI Master thread runs this function, on all ranks
    // When done, this function set done to true and is_done() now returns true
    void test_completion_join();

public:

    /**
     * \brief Creates a Threadpool associated with a communicator
     * 
     * \param[in] n_threads the number of threads
     * \param[in] comm the communicator
     * \param[in] verb the verbosity level. 0 means not printing; > 0 prints more and more to stdout
     * \param[in] basename the prefix to be used to identity this Threadpool in all logging operations
     * \param[in] start_immediately if true, the Threapol starts immediately. Otherwise, the user should call `tp.start()` before `tp.join()`.
     */
    Threadpool_mpi(int n_threads, Communicator *comm, int verb = 0, std::string basename = "Wk_", bool start_immediately = true);

    /**
     * \brief Complete accross all ranks
     * 
     * \details Returns when
     *          1. There are no tasks running or in any queues in all Threadpools associated with the Communicator;
     *          2. There are no messages in flight
     */
    void join();

};

typedef Threadpool_mpi Threadpool;

} // namespace ttor

#endif

#endif