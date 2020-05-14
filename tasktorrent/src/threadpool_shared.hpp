#ifndef __TTOR_SRC_THREADPOOL_SHARED_HPP__
#define __TTOR_SRC_THREADPOOL_SHARED_HPP__

#include <utility>
#include <queue>
#include <mutex>
#include <thread>

#include "util.hpp"
#include "tasks.hpp"

namespace ttor
{

typedef std::priority_queue<Task *, std::vector<Task *>, less_pTask> queueT;

class Threadpool_shared
{
public:
    std::atomic<int> tasks_in_flight; // Counts the number of tasks currently live in the thread pool
    std::atomic<bool> done;           // Used to signal that the threadpool should stop
    // Verbosity level
    // TODO: clarify the intent for the different levels of verbosity
    int verb;
    std::string basename; // Use for debugging, profiling, logging

private:
    // The threads
    std::vector<std::thread> threads;

    // Tasks that can be stolen by other threads
    std::vector<queueT> ready_tasks;
    std::vector<std::mutex> ready_tasks_mtx;

    // Tasks that cannot be stolen
    std::vector<queueT> bound_tasks;
    std::vector<std::mutex> bound_tasks_mtx;

    bool log;
    Logger *logger;

    // Debug only
    std::atomic<int> total_tasks;

    // Run task t
    void consume(int self, Task *t, const std::string &name);

    // test_completion() is overloaded by Threadpool_mpi
    virtual void test_completion();

public:
    /**
     * Creates a threadpool
     * \param n_threads the number of threads in the threadpool
     * \param verb verbosity level. 0 is quiet, > 0 prints more and more informations to stdout
     * \param basename used in logging, this will be used to identity this threadpool
     * \param start_immediately if true, the threadpool starts immediately. Otherwise, use is reponsible for calling `tp.start()` before joining.
     */
    Threadpool_shared(int n_threads, int verb = 0, std::string basename = "Wk_", bool start_immediately = true);

    /**
     * Starts the threadpool
     */
    void start();

    /**
     * Insert task `t`
     * \param t a pointer to the task to insert. The task ownership is taken by the threadpool and the task will be freed when done.
     * \param where the thread to insert the task on
     * \param binding if true, the task cannot be stolen by another thread.
     */
    void insert(Task *t, int where, bool binding = false);

    /**
     * Same as `join()`
     */
    void all_threads_join();

    /**
     * Join all the thread.
     * This function returns when all threads are finished and have joined with the master thread
     */
    void join();

    /**
     * Returns true if no tasks are present in the Taskflow, either running or in the ready queues
     * \return true is all queues are empty and no tasks are running, false otherwise
     */
    bool is_done();

    /**
     * Set the logger
     * \param logger_ a pointer to the `Logger`. Logger ownership is left to the user.
     */
    void set_logger(Logger *logger_);

    /**
     * Returns the number of threads
     * \return the number of threads in the threadpool
     */
    int size();
};

#ifdef TTOR_SHARED

typedef Threadpool_shared Threadpool;

#endif

} // namespace ttor

#endif
