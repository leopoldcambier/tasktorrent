#ifndef __TTOR_RUNTIME_HPP__
#define __TTOR_RUNTIME_HPP__

#include <iostream>
#include <tuple>
#include <utility>
#include <thread>
#include <unordered_map>
#include <array>
#include <deque>
#include <queue>
#include <mutex>
#include <tuple>
#include <memory>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <functional>
#include <limits>

#include "util.hpp"

#ifndef TTOR_SHARED
#include "communications.hpp"
#endif

namespace ttor
{

using std::array;
using std::atomic;
using std::deque;
using std::function;
using std::lock_guard;
using std::make_pair;
using std::make_tuple;
using std::make_unique;
using std::move;
using std::mutex;
using std::pair;
using std::string;
using std::thread;
using std::to_string;
using std::tuple;
using std::unique_ptr;
using std::vector;

template <class T>
struct hash_int_N
{
    size_t operator()(const T &t) const
    {
        return std::hash<T>{}(t);
    }
    bool equal(const T &lhs, const T &rhs) const
    {
        return lhs == rhs;
    }
    size_t hash(const T &h) const
    {
        return (*this)(h);
    }
};

template <class T, std::size_t N>
struct hash_int_N<array<T, N>>
{

    uint32_t get16bits(const char *d)
    {
        return static_cast<uint32_t>((reinterpret_cast<const uint8_t *>(d)[1]) << 8) + static_cast<uint32_t>(reinterpret_cast<const uint8_t *>(d)[0]);
    }

    size_t
    operator()(const array<T, N> &t) const
    {

        // Adapted from https://en.cppreference.com/w/cpp/utility/hash/operator()
        // size_t result = 2166136261;
        // std::hash<T> h;
        // for (size_t i = 0; i < N; i++)
        // {
        //     result = result ^ (h(t[i]) << 1);
        // }
        // return result;

        // https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
        // fnv1a hash
        const uint32_t Prime = 0x01000193; //   16777619
        const uint32_t Seed = 0x811C9DC5;  // 2166136261

        uint32_t hash = Seed;

        const unsigned char *ptr = reinterpret_cast<const unsigned char *>(&t[0]);
        int numBytes = 4 * N;

        while (numBytes--)
            hash = (*ptr++ ^ hash) * Prime;

        return hash;
    }

    bool
    equal(const array<T, N> &lhs, const array<T, N> &rhs) const
    {
        return lhs == rhs;
    }

    size_t hash(const array<T, N> &h) const
    {
        return (*this)(h);
    }
};

struct Task
{
    function<void()> run;
    function<void()> fulfill;
    double priority;
    string name;
    Task() : priority(0), name("_") {}
    const char *c_name()
    {
        return name.c_str();
    }
};

struct less_pTask
{
    bool operator()(const Task *lhs, const Task *rhs)
    {
        return lhs->priority < rhs->priority;
    }
};

typedef std::priority_queue<Task *, vector<Task *>, less_pTask> queueT;

class Threadpool_shared
{
public:
    atomic<int> tasks_in_flight; // Counts the number of tasks currently live in the thread pool
    atomic<bool> done;           // Used to signal that the threadpool should stop
    // Verbosity level
    // TODO: clarify the intent for the different levels of verbosity
    int verb;
    string basename; // Use for debugging, profiling, logging

private:
    // The threads
    vector<thread> threads;

    // Tasks that can be stolen by other threads
    vector<queueT> ready_tasks;
    vector<mutex> ready_tasks_mtx;

    // Tasks that cannot be stolen
    vector<queueT> bound_tasks;
    vector<mutex> bound_tasks_mtx;

    bool log;
    Logger *logger;

    // Debug only
    atomic<int> total_tasks;

public:
    Threadpool_shared(int n_threads, int verb_ = 0, string basename_ = "Wk_", bool start_immediately = true)
        : tasks_in_flight(0),     // number of tasks in flight
          done(false),         // used to signal completion
          verb(verb_), // verbosity level
          basename(basename_),
          threads(n_threads),     // number of threads
          ready_tasks(n_threads), // pool of tasks ready to run
          ready_tasks_mtx(n_threads),
          bound_tasks(n_threads),
          bound_tasks_mtx(n_threads),
          log(false),
          logger(nullptr),
          total_tasks(0) // debug only
    {
        if (start_immediately)
            start();
    }

private:
    // Run task t
    void consume(int self, Task *t, const string &name)
    {
        unique_ptr<Event> e, ef;
        if (verb > 1)
            printf("[%s] %s running\n", name.c_str(), t->c_name());
        if (log)
            e = make_unique<Event>(basename + std::to_string(self) + ">run>" + t->name);

        // run()
        t->run();

        if (log)
            logger->record(move(e));
        if (log)
            ef = make_unique<Event>(basename + std::to_string(self) + ">deps>" + t->name);

        // fulfill()
        t->fulfill();

        if (log)
            logger->record(move(ef));
        if (verb > 1)
            printf("[%s] %s done\n", name.c_str(), t->c_name());

        if (verb > 1)
        {
            ++total_tasks;
            printf("[%s] %s done; total tasks completed %d\n", name.c_str(), t->c_name(), total_tasks.load());
        }

        delete t;

        assert(tasks_in_flight.load() > 0);
        --tasks_in_flight; // The task is complete
    }

    // test_completion() is overloaded by Threadpool_mpi
    virtual void test_completion()
    {
        if (tasks_in_flight.load() != 0)
            return; // quick test to return

        done.store(true);
    }

public:
    void start()
    {
        assert(tasks_in_flight.load() == 0);
        ++tasks_in_flight;
        // This variable will be decremented when we call join().
        // This ensures that the main thread has called fulfill_promise on all indegree-0 tasks
        // before we can terminate the worker threads.

        int n_threads = threads.size();
        for (int self = 0; self < n_threads; self++)
        {
            threads[self] = thread([self, n_threads, this]() {
                // This is the function all threads are continuously running

                string name = basename + std::to_string(self);
                if (verb > 0)
                    printf("[%s] Thread has started\n", name.c_str());

                queueT &rT = ready_tasks[self];
                queueT &bT = bound_tasks[self];
                mutex &rTmtx = ready_tasks_mtx[self];
                mutex &bTmtx = bound_tasks_mtx[self];

                while (true) // Return when join_status == true
                {
                    // Whether we have found a task in the queue
                    bool success = false;
                    Task *t = nullptr; // Task pointer
                    {
                        lock_guard<mutex> lock_rT(rTmtx);
                        lock_guard<mutex> lock_sT(bTmtx);
                        int rT_size = rT.size();
                        int sT_size = bT.size();
                        if (rT_size > 0 && sT_size > 0)
                        {
                            double t_bound_prio = bT.top()->priority;
                            double t_ready_prio = rT.top()->priority;
                            if (t_bound_prio >= t_ready_prio)
                            {
                                t = bT.top();
                                bT.pop();
                            }
                            else
                            {
                                t = rT.top();
                                rT.pop();
                            }
                            success = true;
                        }
                        else if (rT_size > 0)
                        {
                            t = rT.top();
                            rT.pop();
                            success = true;
                        }
                        else if (sT_size > 0)
                        {
                            t = bT.top();
                            bT.pop();
                            success = true;
                        }
                    }

                    if (!success)
                    { // Attempt to steal from ready_tasks
                        unique_ptr<Event> e;
                        for (int j = 1; j < n_threads; j++)
                        {
                            int other = (self + j) % n_threads;
                            queueT &rTother = ready_tasks[other];
                            mutex &rTotherMtx = ready_tasks_mtx[other];
                            {
                                lock_guard<mutex> lock(rTotherMtx);
                                if (rTother.size() > 0)
                                {
                                    t = rTother.top();
                                    rTother.pop();
                                    success = true;

                                    // Log this event
                                    if (log)
                                        e = make_unique<Event>(basename + std::to_string(self) + ">ss>" + basename + std::to_string(other) + ">" + t->name);
                                    if (verb > 1)
                                        printf("[%s] stole task %s from %d\n", name.c_str(), t->c_name(), other);
                                    if (log)
                                        logger->record(move(e));
                                    break;
                                }
                            }
                        }
                    }

                    if (success)
                    {
                        consume(self, t, name);
                        if (verb > 1)
                            printf("[%s] tasks_in_flight %d\n", name.c_str(), tasks_in_flight.load());
                    }
                    else
                    { // no task was found in any queue

                        if(self == 0) {
                            test_completion();
                        }

                        if (is_done())
                        { // Check termination flag
                            if (verb > 1)
                                printf("[%s] Thread is terminating\n", name.c_str());
                            return;
                        }

                        std::this_thread::sleep_for(std::chrono::microseconds(10));
                    }
                }
            });
        };
    }

    void insert(Task *t, int where, bool binding = false)
    {
        if (verb > 1)
            printf("Inserting %s on %d\n", t->c_name(), where);

        if (where < 0 || where >= int(threads.size()))
        {
            printf("Error: where = %d (num threads = %d) for %s\n", where, int(threads.size()), t->c_name());
            assert(false);
        }

        ++tasks_in_flight; // increment task counter

        // binding == false: task is migratable; binding == true: task is bound to thread
        if (!binding)
        {
            lock_guard<mutex> lock(ready_tasks_mtx[where]);
            ready_tasks[where].push(t);
        }
        else
        {
            lock_guard<mutex> lock(bound_tasks_mtx[where]);
            bound_tasks[where].push(t);
        }
    }

    void all_threads_join()
    {
        for (auto &t : threads)
        {
            t.join(); // join all threads
        }
    }

    void join()
    {
        assert(tasks_in_flight.load() > 0);
        // We can safely decrement tasks_in_flight.
        // All tasks have been seeded by the main thread.
        --tasks_in_flight;
        all_threads_join();
    }

    bool is_done()
    {
        return done.load();
    }

    void set_logger(Logger *logger_)
    {
        log = true;
        logger = logger_;
    }

    int size()
    {
        return threads.size();
    }
};

#ifndef TTOR_SHARED

class Threadpool_mpi : public Threadpool_shared
{
private:

    const int my_rank;

    // Global count of all the messages sent and received
    // Used on rank 0
    vector<int>  msgs_queued;       // msgs_queued[from]: user's comm queued rpcs from rank from
    vector<int>  msgs_processed;    // msgs_processed[from]: user's comm processed lpcs from rank from
    vector<int>  tags;              // tags[from]: the greatest ever received confirmation tag

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

    // Last sum
    // Used on rank 0
    int last_sum;  // the last sum_r queued(r) == sum_r processed(r) value
    
    // Used by user to communicate
    Communicator *comm;
    std::string name;

    // Active messages used to determine whether quiescence has been reached
    ActiveMsg<int, int, int>    *am_set_msg_counts_master;   // Send msg count to master
    ActiveMsg<int, int, int>    *am_ask_confirmation;        // Ask worker for confirmation
    ActiveMsg<int, int>         *am_send_confirmation;       // Send confirmation back to master
    ActiveMsg<>                 *am_shutdown_tf;             // Shutdown TF (last message from master to worker)

    // Used to synchronize workers
    // Master increment this every time he sends a new confirmation request
    // Worker reply with the tag
    // We wait for the moment where all workers reply with the latest confirmation_tag, meaning they all confirmed the latest confirmation request
    int confirmation_tag;

    // Update counts on master
    // We use step, provided by the worker, to update msg_queued and msg_processed with the latest information
    void set_msg_counts_master(int from, int msg_queued, int msg_processed) {
        if (verb > 0) {
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
    void ask_confirmation(int msg_queued, int msg_processed, int tag) {
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
    void confirm(int from, int tag) {
        if (verb > 1) {
            printf("[%s] <- %d, Confirmation OK tag %d\n", name.c_str(), from, tag);
        }
        assert(my_rank == 0);
        assert(from >= 0 && from < comm_size());
        tags[from] = std::max(tags[from], tag);
    }

    // Shut down the TF
    void shutdown_tf() {
        if (verb > 0) {
            printf("[%s] Shutting down tf\n", name.c_str());
        }
        assert(tasks_in_flight.load() == 0);
        done.store(true);
    }

    // Everything is done in join
    void test_completion() override {
        // Nothing
    }

public:
    Threadpool_mpi(int n_threads, Communicator *comm_, int verb_ = 0, string basename_ = "Wk_", bool start_immediately = true)
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

    void join()
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

private:

    // Return the number of internal queued rpcs
    int get_intern_n_msg_queued() {
        int nq = comm->get_n_msg_queued() - intern_queued;
        assert(nq >= 0);
        return nq;
    }

    // Return the number of internal processed lpcs
    int get_intern_n_msg_processed() {
        int np = comm->get_n_msg_processed() - intern_processed;
        assert(np >= 0);
        return np;
    }

    // Only MPI Master thread runs this function, on all ranks
    // When done, this function set done to true and is_done() now returns true
    void test_completion_join()
    {

        /**
         * Strategy
         * 
         * We send 4 kinds of messages
         * - Rank !=0 -> Rank 0: latest user rpcs/lpcs counts
         * - Rank   0 -> Rank != 0: when all rpcs/lpcs counts match, sends a 'confirmation request'
         *      We associate 'confirmation request' with a unique tag (or `round`)
         * - Rank !=0 -> Rank 0: reply to the latest 'confirmation request' if the counts still match
         *      The replies use the latest received tag
         * - Rank   0 -> Rank != 0: when all ranks reply to the latest sent confirmation request with their confirmation, we send a shutdown signal
         * 
         * Rank 0 can do two things:
         * - Check reply to the latest confirmation request. If we got a positive reply from all ranks, we send a shutdown message
         * - Otherwise, check the rpcs/lpcs counts. If they all match, we send a confirmation request to all other ranks
         * 
         * Rank != 0 can do two things:
         * - If we have a new rpcs/lpcs count, send the count
         * - If we have a unanswered confirmation request, and if the count haven't changed in the meantime, send a confirmation back to rank 0
         * 
         * Observations:
         * - The internal comms send a finite number of messages, assuming the TF sends a finite number of messages. This solver the fairness issue (no flooding can happen)
         * 
         * - It will terminate. When the TF is empty, all count match, and the confirmation request will be positively replied to
         * 
         * - When the final confirmation request is sent from rank != 0, no more message is sent from rank != 0.
         *      - Previous counts or confirmations arrive on rank 0 before the ultimate confirmation
         *      - Only that final confirmation request can trigger shutdown
         *      - Hence, rank 0 can terminate with a progress-terminate loop
         * - When that shutdown reaches rank != 0, all their message already reached rank 0
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
        if(my_rank == 0) { // Rank 0 test queues/processed global counts
            // STEP A: check the previously received confirmation tags
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
            // If they match, send messages to workers for confirmation
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
};

typedef Threadpool_mpi Threadpool;

#else

typedef Threadpool_shared Threadpool;

#endif

template <class K>
class Taskflow
{

private:
    Threadpool_shared *tp;
    int verb;

    typedef std::unordered_map<K, int, hash_int_N<K>> map_t;
    vector<map_t> dep_map;

    function<void(K)> f_run;
    function<void(K)> f_fulfill;
    function<int(K)> f_mapping;
    function<bool(K)> f_binding;
    function<int(K)> f_indegree;
    function<string(K)> f_name;
    function<double(K)> f_prio;

    // Thread safe
    // Insert task k into dependency map
    void insert(K k)
    {
        int where = -1;
        bool binding;
        auto t = make_task(k, where, binding);
        tp->insert(t, where, binding);
    }

public:
    Taskflow(Threadpool_shared *tp_, int verb_ = 0) : tp(tp_), verb(verb_), dep_map(tp_->size())
    {
        f_name = [](K k) { (void)k; return "_"; };
        f_run = [](K k) { (void)k; printf("Taskflow: undefined task function\n"); };
        f_fulfill = [](K k) { (void)k; };
        f_mapping = [](K k) { (void)k; printf("Taskflow: undefined mapping function\n"); return 0; };
        f_binding = [](K k) { (void)k; return false; /* false = migratable [default]; true = bound to thread */ };
        f_indegree = [](K k) { (void)k; printf("Taskflow: undefined indegree function\n"); return 0; };
        f_prio = [](K k) { (void)k; return 0.0; };
    }

    Task *make_task(K k, int &where, bool &binding)
    {
        Task *t = new Task();
        t->run = [this, k]() { f_run(k); };
        t->fulfill = [this, k]() { f_fulfill(k); };
        t->name = f_name(k);
        t->priority = f_prio(k);
        where = f_mapping(k);
        binding = f_binding(k);
        return t;
    }

    Taskflow &set_task(function<void(K)> f)
    {
        f_run = f;
        return *this;
    }

    Taskflow &set_fulfill(function<void(K)> f)
    {
        f_fulfill = f;
        return *this;
    }

    Taskflow &set_mapping(function<int(K)> f)
    {
        f_mapping = f;
        return *this;
    }

    Taskflow &set_binding(function<int(K)> f)
    {
        f_binding = f;
        return *this;
    }

    Taskflow &set_indegree(function<int(K)> f)
    {
        f_indegree = f;
        return *this;
    }

    Taskflow &set_name(function<string(K)> f)
    {
        f_name = f;
        return *this;
    }

    Taskflow &set_priority(function<double(K)> f)
    {
        f_prio = f;
        return *this;
    }

    string name(K k)
    {
        return f_name(k);
    }

    // Thread-safe
    // Decrease dependency count for task with index k.
    // If task cannot be found in the task map, create a new entry.
    // If it exists, reduce by 1 the dependency count.
    // If count == 0, insert the task in the ready queue.
    void fulfill_promise(K k)
    {
        // Shortcut: if indegree == 1, we can insert the
        // task immediately.
        if (f_indegree(k) == 1)
        {
            insert(k);
            return;
        }

        // We need to create a new entry in the map
        // or decrement the dependency counter.
        const int where = f_mapping(k);
        assert(where >= 0 && where < static_cast<int>(dep_map.size()));

        // Create a task to access and modify the dependency map
        Task *t = new Task();
        t->fulfill = []() {};
        t->name = "dep_map_intern_" + to_string(where);
        t->priority = std::numeric_limits<double>::max();

        t->run = [this, where, k]() {
            auto &dmk = this->dep_map[where];

            auto search = dmk.find(k);
            if (search == dmk.end())
            { // k was not found in the map
                // Insert it
                assert(this->f_indegree(k) > 1);
                auto insert_return = dmk.insert(make_pair(k, this->f_indegree(k)));
                assert(insert_return.second); // (k,indegree) was successfully inserted

                search = insert_return.first; // iterator for key k
            }

            const int count = --search->second; // decrement dependency counter

            if (count < 0)
            {
                printf("Error: count < 0 for %s\n", name(k).c_str());
                assert(false);
            }

            if (verb > 1)
                printf("%s count: %d\n", name(k).c_str(), count);

            if (count == 0)
            {
                // We erase the entry from the map
                dmk.erase(k);
                insert(k);
            }
        };

        tp->insert(t, where, true);
    }
};

} // namespace ttor

#endif
