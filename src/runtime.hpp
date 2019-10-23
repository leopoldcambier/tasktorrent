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
    atomic<int> join_status;     // Used to signal that the threadpool should stop
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
          join_status(0),         // used to signal completion
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

        join_status.store(2); // 2 == signals that the calculation is now complete
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

                        if (self == 0) /* only one worker thread on each rank can do that */
                        {
                            test_completion();
                            // Check whether the DAG has been completely executed and no comms are left.
                            // Send number of msg received and sent to master rank.
                            // Returns immediately if tasks_in_flight != 0.
                            // The master rank will check whether the total number of messages sent and received
                            // matches across all ranks.
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
        --tasks_in_flight;
        // We can safely decrement tasks_in_flight.
        // All tasks have been seeded by the main thread.
        all_threads_join();
    }

    bool is_done()
    {
        return join_status.load() == 2;
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
    vector<int> msg_count_status; // Used to avoid sending multiple completion messages that are identical
    int join_tag;                 // We count the number of rounds to make sure obsolete information is ignored
    mutex join_status_mtx;

    // Multi-rank quiescence
    // Global count of all the messages sent and received
    vector<int> msg_rcvd;
    vector<int> msg_sent;
    // Tracks whether message information from a given rank
    // has been received or not
    vector<bool> msg_rcv_snt_uptd;
    Communicator *comm; // used to communicate

    // Active messages used to determine whether quiescence has been reached
    ActiveMsg<int, int, int, int> *am_notify_master; // use to count messages sent and received from all ranks
    ActiveMsg<int> *am_notify_workers;               // signal all ranks when done

    // This routine is not thread-sage; it needs to be protected by a mutex
    // This is done outside of this routine.
    void join_status_set(int status)
    {
        join_status.store(status); // 0, 1 or 2
        ++join_tag;                // Next round of messages
        if (status == 0)           // Beginning of new round
        {
            msg_count_status[0] = -1; // Reset value
            msg_count_status[1] = -1;
        }
    }

public:
    Threadpool_mpi(int n_threads, Communicator *comm_, int verb_ = 0, string basename_ = "Wk_", bool start_immediately = true)
        : Threadpool_shared(n_threads, verb_, basename_, false),
          msg_count_status(2, -1),       // To avoid sending identical completion messages multiple times
          join_tag(0),                   // Number of completion signals sent
          msg_rcvd(comm_size()),         // number of messages sent
          msg_sent(comm_size()),         // number of messages received
          msg_rcv_snt_uptd(comm_size()), // vector of boolean; whether msg count info has been received or not
          comm(comm_),
          am_notify_master(nullptr), // AM to send msg received and sent
          am_notify_workers(nullptr) // AM to signal completion
    {
        // Call function join_set_msg_counts
        am_notify_master = comm->make_active_msg(
            [&](int &from, int &msg_rcvd, int &msg_sent, int &tag) {
                string name = basename + "MPI_MASTER";
                if (verb > 1)
                    printf("[%s] Receiving message counts (from %d, rcvd %d sent %d)\n", name.c_str(), from, msg_rcvd, msg_sent);
                join_set_msg_counts(from, msg_rcvd, msg_sent, tag);
            });

        // Set join_status to new value
        am_notify_workers = comm->make_active_msg(
            [&](int &status) {
                string name = basename + "MPI_MASTER";
                if (verb > 1)
                    printf("[%s] Receiving status signal %d\n", name.c_str(), status);
                lock_guard<mutex> lock(join_status_mtx);
                join_status_set(status);
            });

        for (int i = 0; i < comm_size(); i++)
            msg_rcv_snt_uptd[i] = false;

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

        while (!is_done() || !comm->is_done())
        {
            comm->progress();
        }

        all_threads_join();
    }

private:
    void test_completion() override
    {
        if (tasks_in_flight.load() != 0)
            return; // quick test to return

        int from = comm_rank();
        int n_msg_rcvd, n_msg_sent, status;
        bool send_counts;
        {
            lock_guard<mutex> lock_recv(comm->recv_count);
            /* This lock is somewhat optional; it prevents the situation where the MPI thread is running 
             join_set_msg_counts() but then stops before incrementing messages_rcvd;
             in that case, when thread worker 0 runs join_set_msg_counts() we are missing one rcvd message.  */

            lock_guard<mutex> lock_status(join_status_mtx);

            // It is critical that we save the number of messages received before
            // checking tasks_in_flight.
            n_msg_rcvd = comm->get_n_msg_rcvd();

            if (tasks_in_flight.load() != 0)
                return;

            status = join_status.load();
            assert(status >= 0 && status <= 2);
            if (status == 2)
                return;
            // The value for status was changed to 2 after calling is_done(); we return and terminate

            n_msg_sent = comm->get_n_msg_sent();
            assert(n_msg_sent >= 0);

            assert(msg_count_status[0] >= -1);
            assert(msg_count_status[1] >= -1);

            // Stage 0: we are updating the number of received messages
            // Stage 1: we are updating the number of sent messages
            bool recv_stage_0 = (status == 0 && msg_count_status[0] < n_msg_rcvd);
            bool send_stage_1 = (status == 1 && msg_count_status[1] < n_msg_sent);

            send_counts = (recv_stage_0 || send_stage_1);

            assert(msg_count_status[0] <= n_msg_rcvd);
            assert(msg_count_status[1] <= n_msg_sent);

            if (send_counts && from != 0)
                ++n_msg_sent;
            // We add 1 here; we are sending a message and it needs to be counted

            // Save the current values for the next call
            if (status == 0)
                msg_count_status[0] = n_msg_rcvd;
            else
                msg_count_status[1] = n_msg_sent;
        }

        if (send_counts)
        {
            // Send message
            string name = basename + std::to_string(0);

            if (from == 0)
            {
                if (verb > 1)
                    printf("[%s] Local rank 0 msg counts update [status %d]: rcvd = %d sent = %d \n", name.c_str(), status, n_msg_rcvd, n_msg_sent);
                join_set_msg_counts(0, n_msg_rcvd, n_msg_sent, join_tag);
            }
            else
            {
                if (verb > 1)
                    printf("[%s] Send msg counts update [status %d]: from %d, rcvd = %d sent = %d\n", name.c_str(), status, from, n_msg_rcvd, n_msg_sent);

                am_notify_master->send(0, from, n_msg_rcvd, n_msg_sent, join_tag);
            }
        }
    }

    // Only rank 0 runs this function
    void join_set_msg_counts(int from, int n_msg_rcvd, int n_msg_sent, int tag)
    {
        assert(comm_rank() == 0);
        assert(from >= 0 && from < comm_size());
        assert(msg_rcvd.size() == comm_size());
        assert(msg_sent.size() == comm_size());
        assert(msg_rcv_snt_uptd.size() == comm_size());
        assert(n_msg_rcvd >= 0);
        assert(n_msg_sent >= 0);

        const bool remote_count(from != 0);

        lock_guard<mutex> lock(join_status_mtx);
        /* Two different threads may be running this function: MPI_MASTER and worker 0.
         * We need a lock to prevent a race condition.
         */

        const int status = join_status.load();
        assert(status >= 0 && status <= 2);

        if (status == 2 /* we are done; just return */ || tag < join_tag /* data is obsolete */)
            return;

        string info = (remote_count ? "MPI_MASTER" : "0");

        const string name = basename + info;

        if (status == 0) // message recv count
        {
            msg_rcvd[from] = n_msg_rcvd;
            msg_rcv_snt_uptd[from] = true;

            if (from == 0)
                // We need to adjust for messages from ranks that we have not received yet
                msg_rcvd[0] += std::count(msg_rcv_snt_uptd.begin(), msg_rcv_snt_uptd.end(), false);

            const bool all_uptd = std::accumulate(msg_rcv_snt_uptd.begin(), msg_rcv_snt_uptd.end(), true, std::logical_and<bool>());

            if (!all_uptd)
                return; // Not all entries have been received

            // Messages that will be sent during stage 1 by all ranks except 0
            msg_rcvd[0] += comm_size() - 1;

            // The message from rank 0 at stage 1 is not counted by the other ranks
            for (int i = 1; i < comm_size(); ++i)
                ++msg_rcvd[i];

            if (verb > 1)
            {
                // Print information about messages received and sent
                for (int i = 0; i < comm_size(); ++i)
                {
                    printf("[%s] Msg count update [stage 0]: from %d rcvd = %d sent = %d \n",
                           name.c_str(), i,
                           msg_rcvd[i],
                           msg_sent[i]);
                }
            }

            // Go on to next stage
            for (int i = 0; i < comm_size(); i++)
                msg_rcv_snt_uptd[i] = false;

            notify_workers(1, info);

            return;
        }
        else if (status == 1) // message sent count
        {
            msg_sent[from] = n_msg_sent;
            msg_rcv_snt_uptd[from] = true;

            const bool all_uptd = std::accumulate(msg_rcv_snt_uptd.begin(), msg_rcv_snt_uptd.end(), true, std::logical_and<bool>());

            if (!all_uptd)
                return; // Not all entries have been received

            const int rcvd_sum = std::accumulate(msg_rcvd.begin(), msg_rcvd.end(), 0, std::plus<int>());
            const int sent_sum = std::accumulate(msg_sent.begin(), msg_sent.end(), 0, std::plus<int>());

            if (verb > 1)
            {
                // Print information about messages received and sent
                for (int i = 0; i < comm_size(); ++i)
                {
                    printf("[%s] Msg count update [stage 1]: from %d rcvd = %d sent = %d \n",
                           name.c_str(), i,
                           msg_rcvd[i],
                           msg_sent[i]);
                }
                printf("[%s] Msg count update: sent - rcvd = %d \n",
                       name.c_str(), sent_sum - rcvd_sum);
            }

            assert(rcvd_sum <= sent_sum);

            if (rcvd_sum == sent_sum)
            {
                // We signal completion; done!
                notify_workers(2, info); // status 2 = dag complete; join()
                return;
            }
            else
            {
                // Go back to stage 0; the number of messages sent does not match the received
                for (int i = 0; i < comm_size(); i++)
                    msg_rcv_snt_uptd[i] = false;

                notify_workers(0, info);
                return;
            }
        }
        else
        {
            assert("Invalid value for status" && false);
        }
    }

    void notify_workers(int status, string info)
    {
        // Only rank 0 can do this
        assert(comm_rank() == 0);
        assert(status >= 0 && status <= 2);

        string name = basename + info;

        if (verb > 1)
            printf("[%s] Signaling all ranks with status %d\n", name.c_str(), status);

        // Send signal to all ranks
        for (int r = 1; r < static_cast<int>(msg_rcvd.size()); ++r)
            am_notify_workers->send(r, status);

        /* We need to be careful when completing; the main thread needs to keep making
           progress on communications before the final join(). */
        join_status_set(status);
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
        t->priority = 0;

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
