#include <fstream>
#include <array>
#include <random>
#include <mutex>
#include <iostream>
#include <map>
#include <memory>

#include "tasktorrent/tasktorrent.hpp"
#include "../common.hpp"

/**
 * n_rows rows of tasks over n_cols columns
 * with n_edges deps between [i,j] and [i+k,j+1] for 0 <= k < n_edges
 */
int wait_chain_deps(const int n_threads, 
                    const int n_rows, 
                    const int n_edges, 
                    const int n_cols, 
                    const double spin_time, 
                    const int repeat, 
                    const int verb) {

    deps_run_repeat("ttor_deps", n_threads, n_rows, n_edges, n_cols, spin_time, repeat, verb, [&](){

        ttor::Threadpool_shared tp(n_threads, verb <= 1 ? 0 : verb, "Wk_");
        ttor::Taskflow<int2> tf(&tp, verb <= 1 ? 0 : verb);
        std::atomic<size_t> n_tasks_ran(0);
        const int n_tasks = n_rows * n_cols;

        tf.set_mapping([&](int2 ij) {
            return (ij[0] % n_threads);
        })
        .set_indegree([&](int2 ij) {
            return (ij[1] == 0 ? 1 : n_edges);
        })
        .set_task([&](int2 ij) {
#ifdef CHECK_NTASKS
            n_tasks_ran++;
#endif
            spin_for_seconds(spin_time);
            if(ij[1] < n_cols-1) {
                for(int k = 0; k < n_edges; k++) {
                    tf.fulfill_promise({ (ij[0] + k) % n_rows, ij[1]+1 });
                }
            }
        })
        .set_priority([&](int2 ij) {
            return n_cols - (double)ij[1];
        })
        .set_name([&](int2 ij){return std::to_string(ij[0]) + "_" + std::to_string(ij[1]);});

        auto t0 = ttor::wctime();
        for(int k = 0; k < n_rows; k++) {
            tf.fulfill_promise({k,0});
        }
        tp.join();
        auto t1 = ttor::wctime();
#ifdef CHECK_NTASKS
        if(n_tasks_ran.load() != n_tasks) { printf("n_tasks_ran is wrong!\n"); exit(1); }
#endif
        return ttor::elapsed(t0, t1);

    });
    
    return 0;
}

int main(int argc, char **argv)
{
    int n_threads = 1;
    int n_rows = 10;
    int n_edges = 10;
    int n_cols = 5;
    double spin_time = 1e-6;
    int verb = 0;
    int repeat = 1;

    if (argc >= 2)
    {
        n_threads = atoi(argv[1]);
        if(n_threads <= 0) { printf("Wrong argument\n"); exit(1); }
    }
    if (argc >= 3)
    {
        n_rows = atoi(argv[2]);
        if(n_rows <= 0) { printf("Wrong argument\n"); exit(1); }
    }
    if (argc >= 4)
    {
        n_edges = atoi(argv[3]);
        if(n_edges <= 0) { printf("Wrong argument\n"); exit(1); }
    }
    if (argc >= 5)
    {
        n_cols = atoi(argv[4]);
        if(n_cols <= 0) { printf("Wrong argument\n"); exit(1); }
    }
    if (argc >= 6)
    {
        spin_time = atof(argv[5]);
        if(spin_time < 0) { printf("Wrong argument\n"); exit(1); }
    }
    if (argc >= 7)
    {
        repeat = atof(argv[6]);
        if(repeat <= 0) { printf("Wrong argument\n"); exit(1); }
    }
    if (argc >= 8)
    {
        verb = atoi(argv[7]);
        if(verb < 0) { printf("Wrong argument\n"); exit(1); }
    }

    if(verb) printf("./ttor_deps n_threads n_rows n_edges n_cols spin_time repeat verb\n");
    int error = wait_chain_deps(n_threads, n_rows, n_edges, n_cols, spin_time, repeat, verb);

    return error;
}
