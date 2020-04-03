#include "communications.hpp"
#include "runtime.hpp"
#include "util.hpp"
#include <fstream>
#include <array>
#include <random>
#include <mutex>
#include <iostream>
#include <map>
#include <memory>

/**
 *   /x\
 *  o-x-y
 *   \x/
 *  
 *   /x\
 *  o-x-y
 *   \x/
 */
int wait_only(const int n_threads, const int n_tasks, const int n_deps, const int sleep_time, const int verb) {

    ttor::Threadpool_shared tp(n_threads, 0, "Wk_", false);
    ttor::Taskflow<int> tf_0(&tp, 0);
    ttor::Taskflow<int> tf_1(&tp, 0);
    ttor::Taskflow<int> tf_2(&tp, 0);

    tf_0.set_mapping([&](int k) {
        return (k % n_threads);
    })
    .set_indegree([&](int k) {
        return 1;
    })
    .set_task([&](int k) {
        std::this_thread::sleep_for(std::chrono::nanoseconds(sleep_time));
        for(int i = 0; i < n_deps; i++) {
            tf_1.fulfill_promise(k * n_deps + i);
        }
    });

    tf_1.set_mapping([&](int k) {
        return ( (k / n_deps) % n_threads);
    })
    .set_indegree([&](int k) {
        return 1;
    })
    .set_task([&](int k) {
        std::this_thread::sleep_for(std::chrono::nanoseconds(sleep_time));
        tf_2.fulfill_promise(k / n_deps);
    });

    tf_2.set_mapping([&](int k) {
        return (k % n_threads);
    })
    .set_indegree([&](int k) {
        return n_deps;
    })
    .set_task([&](int) {
        std::this_thread::sleep_for(std::chrono::nanoseconds(sleep_time));
    });

    for(int k = 0; k < n_tasks; k++) {
        tf_0.fulfill_promise(k);
    }
    auto t0 = ttor::wctime();
    tp.start();
    tp.join();
    auto t1 = ttor::wctime();
    double time = ttor::elapsed(t0, t1);
    if(verb) printf("n_threads,n_taks,n_deps,sleep_time,time,total_tasks,time_per_task\n");
    int total_tasks = n_tasks + n_tasks * n_deps + (n_deps > 0 ? 1 : 0) * n_tasks;
    printf("%d,%d,%d,%d,%e,%d,%e\n", n_threads, n_tasks, n_deps, sleep_time, time, total_tasks, time / total_tasks);
    return 0;
}

int main(int argc, char **argv)
{
    int n_threads = 1;
    int n_tasks = 1000000;
    int n_deps = 0;
    int sleep_time = 1000;
    int verb = 0;

    if (argc >= 2)
    {
        n_threads = atoi(argv[1]);
        assert(n_threads > 0);
    }
    if (argc >= 3)
    {
        n_tasks = atoi(argv[2]);
        assert(n_tasks >= 0);
    }
    if (argc >= 4)
    {
        n_deps = atoi(argv[3]);
        assert(n_deps >= 0);
    }
    if (argc >= 5)
    {
        sleep_time = atoi(argv[4]);
        assert(sleep_time >= 0);
    }
    if (argc >= 6)
    {
        verb = atoi(argv[5]);
        assert(verb >= 0);
    }

    int error = wait_only(n_threads, n_tasks, n_deps, sleep_time, verb);

    return error;
}
