#ifndef __TTOR_SRC_Threadpool_dist_HPP__
#define __TTOR_SRC_Threadpool_dist_HPP__

#if defined(TTOR_MPI) || defined(TTOR_UPCXX)

#include <vector>
#include <string>

#include "threadpool_shared.hpp"
#include "communications.hpp"
#include "communications_mpi.hpp"
#include "communications_upcxx.hpp"

namespace ttor 
{

// Specialized active messages type for threadpool_dist
namespace details {

    template<typename T>
    struct Get_AM {
        typedef void AM_cill;
        typedef void AM_clll;
        typedef void AM_cl;
    };

#if defined(TTOR_MPI)

    template<>
    struct Get_AM<Communicator_MPI> {
        typedef Communicator_MPI::ActiveMsg<char,int,llint,llint>   AM_cill ;
        typedef Communicator_MPI::ActiveMsg<char,llint,llint,llint> AM_clll ;
        typedef Communicator_MPI::ActiveMsg<char,llint>             AM_cl   ;
    };

#elif defined(TTOR_UPCXX)

    template<>
    struct Get_AM<Communicator_UPCXX> {
        typedef Communicator_UPCXX::ActiveMsg<char,int,llint,llint>   AM_cill;
        typedef Communicator_UPCXX::ActiveMsg<char,llint,llint,llint> AM_clll;
        typedef Communicator_UPCXX::ActiveMsg<char,llint>             AM_cl;
    };

#endif

    template<class C> using AM_cill_t = typename Get_AM<C>::AM_cill;
    template<class C> using AM_clll_t = typename Get_AM<C>::AM_clll;
    template<class C> using AM_cl_t   = typename Get_AM<C>::AM_cl;

} // namespace details

/**
 * \brief A threadpool associated to active messages and a communicator.
 * 
 * \details This class behaves exactly like `Threadpool_shared`
 *          except that `join()` is overloaded to complete 
 *          when all threadpools on all ranks have completed
 *          and there are no in-flight active messages 
 *          (i.e. all queued rpcs have been processed).
 */
template<typename Communicator_t>
class Threadpool_dist : public Threadpool_shared
{
    
private:

    const int my_rank;
    const int num_ranks;

    // Global count of all the messages sent and received
    // Used on rank 0
    std::vector<llint>  msgs_queued;       // msgs_queued[from]: user's comm queued rpcs from rank from
    std::vector<llint>  msgs_processed;    // msgs_processed[from]: user's comm processed lpcs from rank from
    std::vector<llint>  tags;              // tags[from]: the greatest ever received confirmation tag

    // Count the number of queued and processed AM's used in the join()
    // Used on all ranks
    llint intern_queued;     // the number of internal queued rpcs
    llint intern_processed;  // the number of internal processed lpcs

    // The last information used/send
    // Used on all ranks except 0
    llint last_sent_nqueued;    // the last sent value of user's queued rpcs
    llint last_sent_nprocessed; // the last sent value of usue's processed lpcs

    // The last confirmaton request and confirmation information
    // Used on all ranks except 0
    llint last_sent_conf_tag;
    llint last_rcvd_conf_tag;
    llint last_rcvd_conf_nqueued;
    llint last_rcvd_conf_nprocessed;

    // Last sum, the last sum_r queued(r) == sum_r processed(r) value checked
    // Used on rank 0
    llint last_sum;  
    
    // Used by user to communicate
    Communicator_t *comm;
    std::string name;

    // Active messages used to determine whether quiescence has been reached
    details::AM_cill_t<Communicator_t>    *am_set_msg_counts_master;   // Send msg count to master
    details::AM_clll_t<Communicator_t>    *am_ask_confirmation;        // Ask worker for confirmation
    details::AM_cill_t<Communicator_t>    *am_send_confirmation;       // Send confirmation back to master
    details::AM_cl_t<Communicator_t>      *am_shutdown_tf;             // Shutdown TF (last message from master to worker)

    // To detect completion
    // Used on rank 0
    llint confirmation_tag;

    // How many internal messages queued from ranks != 0 to rank 0
    // Used on rank 0
    // Gets updated whenever receiving a positive confirmation
    std::vector<llint> intern_msgs_queued_from_nonmaster;

    // How many internal messages queued from rank 0 to tanks != 0
    // Used on rank 0
    std::vector<llint> intern_msgs_queued_to_nonmaster;

    // How many internal messages queued from rank 0 to rank != 0
    // Used on all ranks except 0
    // Gets updated when receiving shutdown message
    llint intern_msgs_queued_from_master;

    // Update counts on master
    // We use step, provided by the worker, to update msg_queued and msg_processed with the latest information
    void set_msg_counts_master(int from, llint msg_queued, llint msg_processed);

    // Ask confirmation on worker
    // If step is the latest information send, and if we're still idle and there were no new messages in between, reply with the tag
    void ask_confirmation(llint msg_queued, llint msg_processed, llint tag);

    // Update tags on master with the latest confirmation tag
    void confirm(int from, llint tag, llint intern_queued_from_nonmaster);

    // Shut down the TF
    void shutdown_tf(llint intern_queued_from_master);

    // Everything is done in join
    void test_completion() override;

    // Return the number of user queued rpcs
    llint get_user_n_msg_queued();

    // Return the number of user processed lpcs
    llint get_user_n_msg_processed();

    // Only MPI Master thread runs this function, on all ranks
    // When done, this function set done to true and is_done() now returns true
    void test_completion_join();

    // Return the expected number of incoming internal messages after completion of user AMs
    llint internal_finish_counts();

public:

    /**
     * \brief Creates a Threadpool associated with a communicator.
     * 
     * \param[in] n_threads the number of threads.
     * \param[in] comm the communicator (either `Communicator_MPI` or `Communicator_UPCXX`)
     * \param[in] verb the verbosity level. 0 means not printing; > 0 prints more and more to stdout.
     * \param[in] basename the prefix to be used to identity `this` in all logging operations.
     * \param[in] start_immediately if true, `this` starts immediately. Otherwise, the user should call `tp.start()` before `tp.join()`.
     * 
     * \pre `n_threads >= 1`
     * \pre `verb >= 0`
     * \pre `comm` points to a valid `Communicator` which should not be destroyed while `this` is in use.
     */
    Threadpool_dist(int n_threads, Communicator_t *comm, int verb = 0, std::string basename = "Wk_", bool start_immediately = true);

    /**
     * \brief Complete accross all ranks
     * 
     * \details Returns when
     *          1. There are no tasks running or in any queues in all Threadpools associated with the Communicator;
     *          2. There are no messages in flight
     * 
     * \pre `this` has been started (through `start()` or the `start_immediately` constructor field).
     * 
     * \post After `tp.join`, 
     *       1. `Threadpool_dist::is_done()` returns `true` on all ranks;
     *       2. `Communicator::is_done()` returns `true` on all ranks;
     *       3. All queued active message on a sender have been processed on a receiver.
     */
    void join();

};

#if defined(TTOR_MPI)

/** 
 * `Threadpool` aliases to `Threadpool_dist<Communicator_MPI>` when `TTOR_MPI` is defined
 */
using Threadpool = Threadpool_dist<Communicator_MPI>;

#elif defined(TTOR_UPCXX)

/** 
 * `Threadpool` aliases to `Threadpool_dist<Communicator_UPCXX>` when `TTOR_UPCXX` is defined
 */
using Threadpool = Threadpool_dist<Communicator_UPCXX>;

#endif

} // namespace ttor

#endif

#endif