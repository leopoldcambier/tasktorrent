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

/**
 * \brief   A set of threads continuously running tasks
 *
 * \details This class represents a set (pool) of threads
 *          Threads in the threadpool continuously run any task given to them
 *          and can in general steal tasks from each other.
 *          Once created, a threadpool is started and run until all queues are empty
 *          and all tasks are done running.
 */
class Threadpool_shared
{

protected:

    std::atomic<int> tasks_in_flight; // Counts the number of tasks currently live in the thread pool
    std::atomic<bool> done;           // Used to signal that the threadpool should stop
    // Verbosity level
    // TODO: clarify the intent for the different levels of verbosity
    const int verb;
    const std::string basename; // Use for debugging, profiling, logging

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

protected:

    // Same as `join()`
    void all_threads_join();

public:

    /**
     * \brief Creates a threadpool with a certain number of threads
     * 
     * \param[in] n_threads the number of threads in the threadpool
     * \param[in] verb verbosity level. 0 is quiet, > 0 prints more and more informations to stdout
     * \param[in] basename used in logging, this will be used to identity this threadpool
     * \param[in] start_immediately if true, the threadpool starts immediately. Otherwise, use is reponsible for calling `tp.start()` before joining.
     */
    Threadpool_shared(int n_threads, int verb = 0, std::string basename = "Wk_", bool start_immediately = true);

    /**
     * \brief Starts the threadpool.
     * 
     * \details Should not be called if the threadpool was created with the option `start_immediately` set to `true`.
     */
    void start();

    /**
     * \brief Insert a task
     * 
     * \details Inserts a task on a given thread. By default the task can be stolen by another thread, unless `binding` is set to `false`.
     * 
     * \param[in] t a pointer to the task to insert. The task ownership is taken by the threadpool and the task will be freed when done.
     * \param[in] where the thread to insert the task on
     * \param[in] binding if true, the task cannot be stolen by another thread.
     */
    void insert(Task *t, int where, bool binding = false);

    /**
     * \brief Join all the threads.
     * 
     * \details This function returns when all threads are finished and have joined with the master thread.
     * All threads are finished when all task queues are empty and no task are currently in the threadpool.
     */
    void join();

    /**
     * \brief Wether the Threadpool is idle or not.
     * 
     * \return true is all queues are empty and no tasks are running, false otherwise
     */
    bool is_done();

    /**
     * \brief Attach a logger
     * 
     * \param[in] logger a pointer to the `Logger`. Logger ownership is left to the user.
     */
    void set_logger(Logger *logger);

    /**
     * \brief The number of threads
     * 
     * \return the number of threads in the threadpool
     */
    int size();
};

#ifdef TTOR_SHARED

typedef Threadpool_shared Threadpool;

#endif

} // namespace ttor

#endif
