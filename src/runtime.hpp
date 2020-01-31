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
    void test_completion()
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
        while(! is_done()) {
            test_completion();
            std::this_thread::sleep_for(std::chrono::microseconds(1000));
        }
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
    // Rank 0-only data
    // Global count of all the messages sent and received
    vector<int>   msgs_queued;
    vector<int>   msgs_processed;
    vector<bool>  confirmations; 

    // All ranks data
    const int my_rank;
    int step;
    int intern_queued;
    int intern_processed;
    bool terminate;
    int conf_msgs_processed;
    int conf_msgs_queued;

    // When to start the steps
    vector<bool>  step_1_ready;
    bool step_2_ready;
    vector<bool>  step_3_ready;
    bool step_4_ready;
    
    // Used by user to communicate
    Communicator *comm;
    std::string name;

    // Active messages used to determine whether quiescence has been reached
    ActiveMsg<int, int, int>    *am_step_0;
    ActiveMsg<int, int>         *am_step_1;
    ActiveMsg<int, bool>        *am_step_2;
    ActiveMsg<bool>             *am_step_3;

    // Update counts on master
    // We use step, provided by the worker, to update msg_queued and msg_processed with the latest information
    void set_msg_counts_master(int from, int nq, int np) {
        assert(comm_rank() == 0);
        assert(from >= 0 && from < comm_size());
        step_1_ready[from] = true;
        msgs_queued[from] = nq;
        msgs_processed[from] = np;
    }

public:
    Threadpool_mpi(int n_threads, Communicator *comm_, int verb_ = 0, string basename_ = "Wk_", bool start_immediately = true)
        : Threadpool_shared(n_threads, verb_, basename_, false),
          // Rank-0 only
          msgs_queued(comm_size(), -2),
          msgs_processed(comm_size(), -2),
          confirmations(comm_size(), false),
          // All ranks
          my_rank(comm_rank()),
          step(0),
          intern_queued(0),
          intern_processed(0),
          terminate(false),
          conf_msgs_processed(-2),
          conf_msgs_queued(-2),
          // When to start steps
          step_1_ready(comm_size(), false),
          step_2_ready(false),
          step_3_ready(comm_size(), false),
          step_4_ready(false),
          // Should we stop
          comm(comm_),
          name(basename + "MPI_MASTER"),
          am_step_0(nullptr),
          am_step_1(nullptr),
          am_step_2(nullptr),
          am_step_3(nullptr)
    {
        // Update message counts on master
        am_step_0 = comm->make_active_msg(
            [&](int &from, int &msg_queued, int &msg_processed) {
                assert(my_rank == 0);
                set_msg_counts_master(from, msg_queued, msg_processed);
                intern_processed++;
            });

        // Ask worker for confirmation on the latest count
        am_step_1 = comm->make_active_msg(
            [&](int &nq, int &np) {
                assert(my_rank != 0);
                conf_msgs_queued = nq;
                conf_msgs_processed = np;
                step_2_ready = 2;
                intern_processed++;
            });

        // Send confirmation to master
        am_step_2 = comm->make_active_msg(
            [&](int& from, bool &result) {
                assert(my_rank == 0);
                step_3_ready[from] = true;
                confirmations[from] = result;
                intern_processed++;
            });

        // Shutdown worker or master
        am_step_3 = comm->make_active_msg(
            [&](bool &all_confirm) {
                terminate = all_confirm;
                step_4_ready = true;
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
                test_completion();
            }
        }
        assert(tasks_in_flight.load() == 0);
        assert(is_done());
        assert(comm->is_done());

        // All threads join
        all_threads_join();
    }

private:

    /** Completion strategy
     * We define 5 steps
     * Rank 0 does steps  0 -> 1 -> 3 -> 4
     * Rank != does steps 0 -> 2 -> 4
     * 
     * Step 0 [All ranks]
     *      - Everyone sends to rank 0 their queued/processed number of USER's rpcs/lpcs (we substract the number of ones used for completion comms)
     *      - When the counts arrive, it makes rank 0 "partially" ready for step 1
     *      - Rank 0 go to step 1
     *      - Rank 2 go to step 2
     * Step 1 [Rank 0 only]
     *      - When rank 0 has received counts from all other ranks, it check all received counts.
     *      - If the counts match, he sends the counts back to other ranks, and make them `ready` for step 2
     *      - If the counts don't match, he sends back (-1, -1) to other ranks, and make them `ready` for step 2
     *      - Rank 0 goes to step 3
     * Step 2 [Rank != 0 only]
     *      - When ranks != 0 is ready, it check the confirmation counts
     *      - It the count still match, he sends back a TRUE to rank 0, making it partially ready for step 3
     *      - If the count don't match, he sends back a FALSE to rank 0, making it partially ready for step 3
     *      - Rank != 0 go to step 4
     * Step 3 [Rank 0 only]
     *      - When rank 0 has received confirmations from all other ranks...
     *      - It check if they are all true.
     *      - He sends that result to all other ranks, making them ready for step 4
     * Step 4 [All ranks]
     *      - When rank 0/!=0 is ready, it checks the result from rank 0
     *      - If the result is terminates, it shuts down
     *      - Otherwise, rank goes to step 0
     **/

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
    // Should only be called when the TF is idle
    void test_completion()
    {
        assert(! is_done());
        assert(tasks_in_flight.load() == 0);
        // Everyone send count to master
        // Master -> 1
        // Worker -> 2
        if(step == 0) {
            int nq = get_intern_n_msg_queued();
            int np = get_intern_n_msg_processed();
            if(my_rank != 0) {
                intern_queued++;
                int from = my_rank;
                am_step_0->send(0, from, nq, np);
                // Go to step 2
                step = 2;
            } else {
                set_msg_counts_master(0, nq, np);
                // Go to step 1
                step = 1;
                step_1_ready[0] = true;
            }
        // Master receives and check counts. 
        // Master -> 3
        } else if (step == 1) {
            assert(my_rank == 0);
            const bool ready = std::all_of(step_1_ready.begin(), step_1_ready.end(), [](bool b){return b;});
            if(ready) {
                const int  queued_sum    = std::accumulate(msgs_queued.begin(),    msgs_queued.end(), 0, std::plus<int>());
                const int  processed_sum = std::accumulate(msgs_processed.begin(), msgs_processed.end(), 0, std::plus<int>());
                bool maybe_terminate = (queued_sum == processed_sum);
                for(int r = 1; r < comm_size(); r++) {
                    int nq, np;
                    if(maybe_terminate) {
                        nq = msgs_queued[r];
                        np = msgs_processed[r];
                    } else {
                        nq = -1;
                        np = -1;
                    }
                    intern_queued++;
                    am_step_1->send(r, nq, np);
                }
                confirmations[0] = maybe_terminate;
                std::fill(step_1_ready.begin(), step_1_ready.end(), false);
                std::fill(msgs_queued.begin(), msgs_queued.end(), -2);
                std::fill(msgs_processed.begin(), msgs_processed.end(), -2);
                // Go to step 3 next
                step = 3;
                // Rank 0 isn't ready yet, it needs all the am_step_2 to be processed
                step_3_ready[0] = true;
            }
        // Worker checks received count
        // Sends back count to 0
        // -> 3
        } else if (step == 2) {
            assert(my_rank != 0);
            if(step_2_ready) {
                int nq = get_intern_n_msg_queued();
                int np = get_intern_n_msg_processed();
                bool reply = (conf_msgs_queued == nq && conf_msgs_processed == np);
                intern_queued++;
                int from = my_rank;
                am_step_2->send(0, from, reply);
                step_2_ready = false;
                // Go to step 4 next
                // Rank != 0 isn't ready for that yet, and need the result from am_step_3 to be processed
                step = 4;
            }
        // 0 check if all count still match
        // If yes -> 4
        // If not -> 0
        } else if (step == 3) {
            assert(my_rank == 0);
            const bool ready = std::all_of(step_3_ready.begin(), step_3_ready.end(), [](bool b){return b;});
            if(ready) {
                bool all_confirm = std::all_of(confirmations.begin(), confirmations.end(), [](bool b){return b;});
                for(int r = 1; r < comm_size(); r++) {
                    intern_queued++;
                    am_step_3->send(r, all_confirm);
                }
                terminate = all_confirm;
                std::fill(step_3_ready.begin(), step_3_ready.end(), false);
                std::fill(confirmations.begin(), confirmations.end(), false);
                // What do we do next
                step = 4;
                // Rank 0 can go right now
                // Other need am_step_3 to be processed to move forward
                step_4_ready = true;
            }
        // All check `terminate`
        // If true, we stop the TF
        // Else, go to 0
        } else if (step == 4) {
            if(step_4_ready) {
                if(terminate) {
                    done.store(true);
                } else {
                    step = 0;
                }
                terminate = false;
                step_4_ready = false;
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
