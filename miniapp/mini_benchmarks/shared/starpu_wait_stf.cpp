#include <starpu.h>
#include <fstream>
#include <array>
#include <random>
#include <mutex>
#include <iostream>
#include <map>
#include <memory>
#include <atomic>

#include "common.hpp"

/** 
 * Compile with something like
 * icpc -I${HOME}/Softwares/hwloc-2.2.0/install/include -I${HOME}/Softwares/starpu-1.3.2/install/include/starpu/1.3 -lpthread -L${HOME}/Softwares/hwloc-2.2.0/install/lib -L${HOME}/Softwares/starpu-1.3.2/install/lib  -lstarpu-1.3 -lhwloc starpu_wait.cpp -O3 -o starpu_wait_stf -Wall
 */

double SPIN_TIME = 0.0;
std::atomic<size_t> n_tasks_ran(0);

void task(void *buffers[], void *cl_arg) { 
#ifdef CHECK_NTASKS
    n_tasks_ran++;
#endif
    spin_for_seconds(SPIN_TIME);
}

struct starpu_codelet task_cl = {
    .where = STARPU_CPU,
    .cpu_funcs = { task, NULL },
    .nbuffers = 1,
    .modes = { STARPU_RW }
};

int wait_only(const int n_tasks, const double spin_time, const int repeat, const int verb) {

    SPIN_TIME = spin_time;
    const int n_threads = get_starpu_num_cores();

    wait_only_run_repeat("startpu_wait_stf", n_threads, n_tasks, spin_time, repeat, verb, [&](){

        n_tasks_ran.store(0);
        int err = starpu_init(NULL);
        if(err != 0) { printf("Error in starpu_init!\n"); exit(1); }
        auto data = std::vector<int>(n_tasks, 0);
        auto handles = std::vector<starpu_data_handle_t>(n_tasks);
        for(int k = 0; k < n_tasks; k++) {
            starpu_variable_data_register(&handles[k], STARPU_MAIN_RAM, (uintptr_t)(&data[k]), sizeof(int));
        }

        const auto t0 = wtime_now();
        for(int k = 0; k < n_tasks; k++) {
            int ret = starpu_task_insert(&task_cl, STARPU_RW, handles[k], 0);
            if(err != 0) { printf("Error in starpu_task_insert!\n"); exit(1); }
        }
        starpu_task_wait_for_all();
        const auto t1 = wtime_now();

        for(int k = 0; k < n_tasks; k++) {
            starpu_data_unregister(handles[k]);
        }
        starpu_shutdown();
    
#ifdef CHECK_NTASKS
        if(n_tasks_ran.load() != n_tasks) { printf("n_tasks_ran is wrong!\n"); exit(1); }
#endif
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

    if(verb) printf("STARPU_NCPU=XX ./starpu_wait_stf n_tasks spin_time verb\n");
    int error = wait_only(n_tasks, spin_time, repeat, verb);
    return error;
}
