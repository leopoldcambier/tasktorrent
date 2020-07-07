#include <fstream>
#include <iostream>
#include "common.hpp"
#include <omp.h>

/**
 * Run n_tasks that only spins for a certain amount of time
 * Compile with something like 
 * icpc -qopenmp -O3 omp_wait.cpp -o omp_wait
 * g++ -fopenmp -O3 omp_wait.cpp -o omp_wait
 */
int wait_only(const int n_tasks, const double spin_time, const int repeat, const int verb) {

    const int n_threads = omp_get_max_threads();

    wait_only_run_repeat("omp_wait", n_threads, n_tasks, spin_time, repeat, verb, [&](){

        const auto t0 = wtime_now();
        #pragma omp parallel
        {
            #pragma omp single nowait
            {
                for(int i = 0; i < n_tasks; i++) {
                    #pragma omp task
                    {
                        spin_for_seconds(spin_time);
                    }
                }
                #pragma omp taskwait
            }
        }
        const auto t1 = wtime_now();
        return wtime_elapsed(t0, t1);

    });

    return 0;
}

int main(int argc, char **argv)
{
    int n_tasks = 1000;
    double spin_time = 1e-6;
    int repeat = 1;
    int verb = 0;

    if (argc >= 2)
    {
        n_tasks = atoi(argv[1]);
        if(n_tasks < 0) { printf("Wrong argument\n"); exit(1); }
    }
    if (argc >= 3)
    {
        spin_time = atof(argv[2]);
        if(spin_time < 0) { printf("Wrong argument\n"); exit(1); }
    }
    if (argc >= 4)
    {
        repeat = atof(argv[3]);
        if(repeat <= 0) { printf("Wrong argument\n"); exit(1); }
    }
    if (argc >= 5)
    {
        verb = atoi(argv[4]);
        if(verb < 0) { printf("Wrong argument\n"); exit(1); }
    }

    if(verb) printf("OMP_NUM_THREADS=16 ./omp_wait n_tasks spin_time verb\n");
    int error = wait_only(n_tasks, spin_time, repeat, verb);

    return error;
}
