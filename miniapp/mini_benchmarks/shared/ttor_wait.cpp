#include <fstream>
#include <array>
#include <random>
#include <mutex>
#include <iostream>
#include <map>
#include <memory>

#include "../common.hpp"
#include "tasktorrent/tasktorrent.hpp"

/**
 * Run n_tasks that only spins for a certain amount of time
 */
int wait_only(const int n_threads, const int n_tasks, const double spin_time, const bool time_insertion, const int repeat, const int verb) {

    wait_only_run_repeat("ttor_wait_time_insertion" + std::to_string(time_insertion), n_threads, n_tasks, spin_time, repeat, verb, [&](){

        ttor::Threadpool_shared tp(n_threads, 0, "Wk_", false);
        ttor::Taskflow<int> tf(&tp, 0);

        tf.set_mapping([&](int k) {
            return (k % n_threads);
        })
        .set_indegree([&](int k) {
            return 1;
        })
        .set_task([&](int k) {
            spin_for_seconds(spin_time);
        });

        double time = 0.0;
        if(time_insertion) { // For comparison with *PU and OpenMP
            const auto t0 = wtime_now();
            tp.start();
            for(int k = 0; k < n_tasks; k++) {
                tf.fulfill_promise(k);
            }
            tp.join();
            const auto t1 = wtime_now();
            time = wtime_elapsed(t0, t1);
        } else { // For measuring the "pure serial" overhead
            for(int k = 0; k < n_tasks; k++) {
                tf.fulfill_promise(k);
            }
            const auto t0 = wtime_now();
            tp.start();
            tp.join();
            const auto t1 = wtime_now();
            time = wtime_elapsed(t0, t1);
        }
        return time;
        
    });

    return 0;
}

int main(int argc, char **argv)
{
    int n_threads = 1;
    int n_tasks = 1000;
    double spin_time = 1e-6;
    bool time_insertion = true; // true means we also measure the insertion of all the tasks
    int repeat = 1;
    int verb = 0;

    if (argc >= 2)
    {
        n_threads = atoi(argv[1]);
        if(n_threads <= 0) { printf("Wrong argument\n"); exit(1); }
    }
    if (argc >= 3)
    {
        n_tasks = atoi(argv[2]);
        if(n_tasks < 0) { printf("Wrong argument\n"); exit(1); }
    }
    if (argc >= 4)
    {
        spin_time = atof(argv[3]);
        if(spin_time < 0) { printf("Wrong argument\n"); exit(1); }
    }
    if (argc >= 5)
    {
        time_insertion = (bool)atoi(argv[4]);
    }
    if (argc >= 6)
    {
        repeat = atoi(argv[5]);
        if(repeat <= 0) { printf("Wrong argument\n"); exit(1); }
    }
    if (argc >= 7)
    {
        verb = atoi(argv[6]);
        if(verb < 0) { printf("Wrong argument\n"); exit(1); }
    }

    if(verb) printf("./micro_wait n_threads n_tasks spin_time time_insertion repeat verb\n");
    int error = wait_only(n_threads, n_tasks, spin_time, time_insertion, repeat, verb);

    return error;
}
