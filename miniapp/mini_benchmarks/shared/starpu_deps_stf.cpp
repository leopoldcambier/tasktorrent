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
 * icpc -I${HOME}/Softwares/hwloc-2.2.0/install/include -I${HOME}/Softwares/starpu-1.3.2/install/include/starpu/1.3 -lpthread -L${HOME}/Softwares/hwloc-2.2.0/install/lib -L${HOME}/Softwares/starpu-1.3.2/install/lib  -lstarpu-1.3 -lhwloc starpu_deps.cpp -O3 -o starpu_deps -Wall
 */

// TODO: fix global variable here
double SPIN_TIME = 0.0;
std::atomic<size_t> n_tasks_ran(0);
std::vector<bool> done;

void task(void *buffers[], void *cl_arg) { 
#ifdef CHECK_NTASKS
    n_tasks_ran++;
#endif
    spin_for_seconds(SPIN_TIME);
}

struct starpu_codelet cl_1 =
{
    .cpu_funcs = {task, NULL},
    .nbuffers = 1
};
struct starpu_codelet cl_2 =
{
    .cpu_funcs = {task, NULL},
    .nbuffers = 2
};
struct starpu_codelet cl_3 =
{
    .cpu_funcs = {task, NULL},
    .nbuffers = 3
};
struct starpu_codelet cl_5 =
{
    .cpu_funcs = {task, NULL},
    .nbuffers = 5
};
struct starpu_codelet cl_9 =
{
    .cpu_funcs = {task, NULL},
    .nbuffers = 9
};
struct starpu_codelet cl_17 =
{
    .cpu_funcs = {task, NULL},
    .nbuffers = 17
};
struct starpu_codelet cl_33 =
{
    .cpu_funcs = {task, NULL},
    .nbuffers = 33
};


int wait_chain_deps(const int n_rows, 
                    const int n_edges, 
                    const int n_cols, 
                    const double spin_time, 
                    const int repeat, 
                    const int verb) {

    int n_tasks = n_rows * n_cols;
    const int n_threads = get_starpu_num_cores();
    SPIN_TIME = spin_time;
    
    std::map<int,struct starpu_codelet*> clts;
    clts[1] = &cl_1;
    clts[2] = &cl_2;
    clts[3] = &cl_3;
    clts[5] = &cl_5;
    clts[9] = &cl_9;
    clts[17] = &cl_17;
    clts[33] = &cl_33;

    deps_run_repeat("starpu_deps_stf", n_threads, n_rows, n_edges, n_cols, spin_time, repeat, verb, [&](){

        n_tasks_ran.store(0);
        int err = starpu_init(NULL);
        assert(err == 0);

        auto data = std::vector<int>(n_rows * n_cols);
        auto handles = std::vector<starpu_data_handle_t>(n_rows * n_cols);
        for(int k = 0; k < n_rows * n_cols; k++) {
            starpu_variable_data_register(&handles[k], STARPU_MAIN_RAM, (uintptr_t)(&data[k]), sizeof(int));
        }

        double start = starpu_timing_now();
        for(int j = 0; j < n_cols; j++) {
            for(int i = 0; i < n_rows; i++) {
                // Create list of dependencies
                const int n_deps = (j == 0 ? 0 : n_edges);
                struct starpu_data_descr *descrs = (struct starpu_data_descr*) malloc((n_deps + 1) * sizeof(struct starpu_data_descr));
                // Output
                descrs[0].handle = handles[i+j*n_rows];
                descrs[0].mode = STARPU_RW;
                // Input
                for(int k = 0 ; k < n_deps ; k++) {
                    int i_before = n_rows - ( ( (n_rows - i - 1) + k ) % n_rows ) - 1;
                    descrs[k+1].handle = handles[i_before + (j-1)*n_rows];
                    descrs[k+1].mode = STARPU_R;
                }
                // Submit
                int ret = starpu_task_insert(clts.at(n_deps + 1), STARPU_DATA_MODE_ARRAY, descrs, n_deps + 1, 0);
                assert(ret == 0);
            }
        }
        starpu_task_wait_for_all();
        double end = starpu_timing_now();
        for(int k = 0; k < n_rows * n_cols; k++) {
            starpu_data_unregister(handles[k]);
        }
        starpu_shutdown();
#ifdef CHECK_NTASKS
        if(n_tasks_ran.load() != n_tasks) { printf("n_tasks_ran is wrong!\n"); exit(1); }
#endif
        return (end - start)/1e6;

    });

    return 0;
}

int main(int argc, char **argv)
{
    int n_rows = 10;
    int n_edges = 10;
    int n_cols = 5;
    double spin_time = 1e-6;
    int repeat = 1;
    int verb = 0;

    if (argc >= 2)
    {
        n_rows = atoi(argv[1]);
        if(n_rows <= 0) { printf("Wrong argument\n"); exit(1); }
    }
    if (argc >= 3)
    {
        n_edges = atoi(argv[2]);
        if(n_edges <= 0) { printf("Wrong argument\n"); exit(1); }
    }
    if (argc >= 4)
    {
        n_cols = atoi(argv[3]);
        if(n_cols <= 0) { printf("Wrong argument\n"); exit(1); }
    }
    if (argc >= 5)
    {
        spin_time = atof(argv[4]);
        if(spin_time < 0) { printf("Wrong argument\n"); exit(1); }
    }
    if (argc >= 6)
    {
        repeat = atof(argv[5]);
        if(repeat <= 0) { printf("Wrong argument\n"); exit(1); }
    }
    if (argc >= 7)
    {
        verb = atoi(argv[6]);
        if(verb < 0) { printf("Wrong argument\n"); exit(1); }
    }

    if(verb) printf("./starpu_deps_stf n_rows n_edges n_cols spin_time repeat verb\n");
    int error = wait_chain_deps(n_rows, n_edges, n_cols, spin_time, repeat, verb);

    return error;
}
