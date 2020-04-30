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
    Threadpool_shared(int n_threads, int verb_ = 0, std::string basename_ = "Wk_", bool start_immediately = true);

    void start();

    void insert(Task *t, int where, bool binding = false);

    void all_threads_join();

    void join();

    bool is_done();

    void set_logger(Logger *logger_);

    int size();
};

#ifdef TTOR_SHARED

typedef Threadpool_shared Threadpool;

#endif

} // namespace ttor

#endif
