#include "tasktorrent/tasktorrent.hpp"
#include <fstream>
#include <array>
#include <random>
#include <mutex>
#include <iostream>
#include <map>
#include <memory>

void spin_for(double time) {
    auto t0 = ttor::wctime();
    while(true) {
    	auto t1 = ttor::wctime();
        if( ttor::elapsed(t0, t1) >= time ) break;
    }
}

typedef std::array<int,2> int2;

/**
 * n_rows rows of tasks over n_cols columns
 * with n_edges deps between [i,j] and [i+k,j+1] for 0 <= k < n_edges
 */
int wait_chain_deps(const int n_threads, const int n_rows, const int n_edges, const int n_cols, const double sleep_for, const int verb) {

    ttor::Threadpool_shared tp(n_threads, verb <= 1 ? 0 : verb, "Wk_", false);
    ttor::Taskflow<int2> tf(&tp, verb <= 1 ? 0 : verb);
    std::atomic<size_t> n_tasks(0);

    tf.set_mapping([&](int2 ij) {
        return (ij[0] % n_threads);
    })
    .set_indegree([&](int2 ij) {
        return (ij[1] == 0 ? 1 : n_edges);
    })
    .set_task([&](int2 ij) {
        n_tasks++;
        spin_for(sleep_for);
        if(ij[1] < n_cols-1) {
            for(int k = 0; k < n_edges; k++) {
                tf.fulfill_promise({ (ij[0] + k) % n_rows, ij[1]+1 });
            }
        }
    })
    .set_name([&](int2 ij){return std::to_string(ij[0]) + "_" + std::to_string(ij[1]);});

    for(int k = 0; k < n_rows; k++) {
        tf.fulfill_promise({k,0});
    }
    auto t0 = ttor::wctime();
    tp.start();
    tp.join();
    auto t1 = ttor::wctime();
    double time = ttor::elapsed(t0, t1);
    if(verb) printf("n_threads,n_rows,n_edges,n_cols,sleep_for,time,total_tasks,efficiency\n");
    int total_tasks = n_rows * n_cols;
    assert(n_tasks.load() == total_tasks);
    double speedup = (double)(total_tasks) * (double)(sleep_for) / (double)(time);
    double efficiency = speedup / (double)(n_threads);
    printf("%d,%d,%d,%d,%e,%e,%d,%e\n", n_threads, n_rows, n_edges, n_cols, sleep_for, time, total_tasks, efficiency);

    return 0;
}

int main(int argc, char **argv)
{
    int n_threads = 1;
    int n_rows = 10;
    int n_edges = 10;
    int n_cols = 5;
    double sleep_for = 1e-6;
    int verb = 0;

    if (argc >= 2)
    {
        n_threads = atoi(argv[1]);
        assert(n_threads > 0);
    }
    if (argc >= 3)
    {
        n_rows = atoi(argv[2]);
        assert(n_rows >= 0);
    }
    if (argc >= 4)
    {
        n_edges = atoi(argv[3]);
        assert(n_edges >= 0);
    }
    if (argc >= 5)
    {
        n_cols = atoi(argv[4]);
        assert(n_cols >= 0);
    }
    if (argc >= 6)
    {
        sleep_for = atof(argv[5]);
        assert(sleep_for >= 0);
    }
    if (argc >= 7)
    {
        verb = atoi(argv[6]);
        assert(verb >= 0);
    }

    if(verb) printf("./micro_wait n_threads n_rows n_edges n_cols sleep_for verb\n");
    int error = wait_chain_deps(n_threads, n_rows, n_edges, n_cols, sleep_for, verb);

    return error;
}
