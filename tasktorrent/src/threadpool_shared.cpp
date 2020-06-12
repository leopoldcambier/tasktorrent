#include <utility>
#include <thread>
#include <queue>
#include <mutex>
#include <memory>
#include <cassert>

#include "util.hpp"
#include "tasks.hpp"
#include "threadpool_shared.hpp"

namespace ttor
{

typedef std::priority_queue<Task *, std::vector<Task *>, less_pTask> queueT;

Threadpool_shared::Threadpool_shared(int n_threads, int verb_, std::string basename_, bool start_immediately)
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

// Run task t
void Threadpool_shared::consume(int self, Task *t, const std::string &name)
{
    std::unique_ptr<Event> e, ef;
    if (verb > 1)
        printf("[%s] %s running\n", name.c_str(), t->c_name());
    if (log)
        e = std::make_unique<Event>(basename + std::to_string(self) + ">run>" + t->name);

    // run()
    t->run();

    if (log)
        logger->record(move(e));
    if (log)
        ef = std::make_unique<Event>(basename + std::to_string(self) + ">deps>" + t->name);

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
void Threadpool_shared::test_completion()
{
    if (tasks_in_flight.load() != 0)
        return; // quick test to return

    done.store(true);
}

void Threadpool_shared::start()
{
    ++tasks_in_flight;
    // This variable will be decremented when we call join().
    // This ensures that the main thread has called fulfill_promise on all indegree-0 tasks
    // before we can terminate the worker threads.

    int n_threads = threads.size();
    for (int self = 0; self < n_threads; self++)
    {
        threads[self] = std::thread([self, n_threads, this]() {

            // This is the function all threads are continuously running

            std::string name = basename + std::to_string(self);
            if (verb > 0)
                printf("[%s] Thread has started\n", name.c_str());

            queueT &rT = ready_tasks[self];
            queueT &bT = bound_tasks[self];
            std::mutex &rTmtx = ready_tasks_mtx[self];
            std::mutex &bTmtx = bound_tasks_mtx[self];

            while (true) // Return when join_status == true
            {
                // Whether we have found a task in the queue
                bool success = false;
                Task *t = nullptr; // Task pointer
                {
                    std::lock_guard<std::mutex> lock_rT(rTmtx);
                    std::lock_guard<std::mutex> lock_sT(bTmtx);
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
                    std::unique_ptr<Event> e;
                    for (int j = 1; j < n_threads; j++)
                    {
                        int other = (self + j) % n_threads;
                        queueT &rTother = ready_tasks[other];
                        std::mutex &rTotherMtx = ready_tasks_mtx[other];
                        {
                            std::lock_guard<std::mutex> lock(rTotherMtx);
                            if (rTother.size() > 0)
                            {
                                t = rTother.top();
                                rTother.pop();
                                success = true;

                                // Log this event
                                if (log)
                                    e = std::make_unique<Event>(basename + std::to_string(self) + ">ss>" + basename + std::to_string(other) + ">" + t->name);
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

void Threadpool_shared::insert(Task *t, int where, bool binding)
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
        std::lock_guard<std::mutex> lock(ready_tasks_mtx[where]);
        ready_tasks[where].push(t);
    }
    else
    {
        std::lock_guard<std::mutex> lock(bound_tasks_mtx[where]);
        bound_tasks[where].push(t);
    }
}

void Threadpool_shared::all_threads_join()
{
    for (auto &t : threads)
    {
        t.join(); // join all threads
    }
}

void Threadpool_shared::join()
{
    assert(tasks_in_flight.load() > 0);
    // We can safely decrement tasks_in_flight.
    // All tasks have been seeded by the main thread.
    --tasks_in_flight;
    all_threads_join();
}

bool Threadpool_shared::is_done()
{
    return done.load();
}

void Threadpool_shared::set_logger(Logger *logger_)
{
    log = true;
    logger = logger_;
}

int Threadpool_shared::size()
{
    return threads.size();
}

} // namespace ttor
