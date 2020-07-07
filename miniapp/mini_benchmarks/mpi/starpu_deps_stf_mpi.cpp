#include <starpu.h>
#include <starpu_mpi.h>
#include <fstream>
#include <array>
#include <random>
#include <mutex>
#include <iostream>
#include <map>
#include <memory>
#include <atomic>
#include <mpi.h>

#include "../common.hpp"

/** 
 * Compile with something like
 * mpiicpc -I${HOME}/Softwares/hwloc-2.2.0/install/include -I${HOME}/Softwares/starpu-1.3.2/install/include/starpu/1.3 -lpthread -L${HOME}/Softwares/hwloc-2.2.0/install/lib -L${HOME}/Softwares/starpu-1.3.2/install/lib  -lstarpumpi-1.3 -lstarpu-1.3 -lhwloc starpu_deps.cpp -O3 -o starpu_deps -Wall
 */

// TODO: fix global variable here
double SPIN_TIME = 0.0;
std::atomic<size_t> n_tasks_ran(0);
std::vector<bool> done;

void task(void *buffers[], void *cl_arg) { 
    n_tasks_ran++;
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

    std::vector<double> efficiencies;
    std::vector<double> times;
    int n_tasks = n_rows * n_cols;

    const char* env_n_cores = std::getenv("STARPU_NCPU");
    assert(env_n_cores != nullptr);
    const int n_threads = atoi(env_n_cores);
    
    SPIN_TIME = spin_time;

    std::map<int,struct starpu_codelet*> clts;
    clts[1] = &cl_1;
    clts[2] = &cl_2;
    clts[3] = &cl_3;
    clts[5] = &cl_5;
    clts[9] = &cl_9;
    clts[17] = &cl_17;
    clts[33] = &cl_33;

    int my_rank, nranks;
    starpu_mpi_comm_rank(MPI_COMM_WORLD, &my_rank);
    starpu_mpi_comm_size(MPI_COMM_WORLD, &nranks);
    const int rows_per_rank = (n_rows + nranks - 1)/nranks;
    const int rows_my_rank = std::min(rows_per_rank, n_rows - my_rank * rows_per_rank);
    auto task_2_rank = [rows_per_rank](int2 ij){ return ij[0] / rows_per_rank; };

    for(int step = 0; step < repeat; step++) {

        n_tasks_ran.store(0);
        int err = starpu_init(NULL);
        assert(err == 0);

        auto data = std::vector<int>(n_rows * n_cols, 0);
        auto handles = std::vector<starpu_data_handle_t>(n_rows * n_cols);
        for(int j = 0; j < n_cols; j++) {
            for(int i = 0; i < n_rows; i++) {
                int k = i + j*n_rows;
                if (my_rank == task_2_rank({i,j})) {
                    starpu_variable_data_register(&handles[k], STARPU_MAIN_RAM, (uintptr_t)(&data[k]), sizeof(int));
                    printf("Registring local %d %d at %d = %p\n", i, j, k, &handles[k]);
                } else {
                    starpu_variable_data_register(&handles[k], -1, (uintptr_t)(nullptr), sizeof(int));
                    printf("Registring remote %d %d at %d = %p\n", i, j, k, &handles[k]);
                }
                if (handles[k]) {
                    starpu_mpi_data_register(handles[k], k, my_rank);
                }
            }
        }

        starpu_mpi_barrier(MPI_COMM_WORLD);
        double start = starpu_timing_now();
        for(int j = 0; j < n_cols; j++) {
            for(int i = 0; i < n_rows; i++) {
                // Create list of dependencies
                const int n_deps = (j == 0 ? 0 : n_edges);
                struct starpu_data_descr *descrs = (struct starpu_data_descr*) malloc((n_deps + 1) * sizeof(struct starpu_data_descr));
                // Output
                printf("(%d %d) using RW (%d %d %d = %p)\n", i, j, i, j, i+j*n_rows, &handles[i+j*n_rows]);
                descrs[0].handle = handles[i+j*n_rows];
                descrs[0].mode = STARPU_RW;
                // Input
                for(int k = 0 ; k < n_deps ; k++) {
                    int i_before = n_rows - ( ( (n_rows - i - 1) + k ) % n_rows ) - 1;
                    printf("(%d %d) using R (%d %d %d = %p)\n", i, j, i_before, j-1, i_before + (j-1)*n_rows, &handles[i_before + (j-1)*n_rows]);
                    descrs[k+1].handle = handles[i_before + (j-1)*n_rows];
                    descrs[k+1].mode = STARPU_R;
                }
                printf("OK!\n");
                // Submit
                int ret = starpu_mpi_task_insert(MPI_COMM_WORLD,clts.at(n_deps + 1), STARPU_DATA_MODE_ARRAY, descrs, n_deps + 1, 0);
                assert(ret == 0);
            }
        }
        starpu_task_wait_for_all();
        starpu_mpi_barrier(MPI_COMM_WORLD);
        double end = starpu_timing_now();
        for(int k = 0; k < n_rows * n_cols; k++) {
            starpu_data_unregister(handles[k]);
        }
        starpu_shutdown();
        double time = (end - start)/1e6;
        if(verb) printf("[my_rank] iteration my_rank nranks repeat n_threads n_rows n_edges n_cols spin_time time n_tasks efficiency\n");
        assert(n_tasks_ran.load() == rows_my_rank * n_cols);
        double speedup = (double)(n_tasks) * (double)(spin_time) / (double)(time);
        double efficiency = speedup / (double)(n_threads);
        efficiencies.push_back(efficiency);
        times.push_back(time);
        printf("[%d]++++ starpudepsstfmpi %d %d %d %d %d %d %d %d %e %e %d %e\n", my_rank, step, my_rank, nranks, repeat, n_threads, n_rows, n_edges, n_cols, spin_time, time, n_tasks, efficiency);
    }

    double eff_mean, eff_std, time_mean, time_std;
    compute_stats(efficiencies, &eff_mean, &eff_std);
    compute_stats(times, &time_mean, &time_std);
    if(verb) printf("[my_rank] my_rank nranks repeat n_threads n_rows n_edges n_cols spin_time n_tasks efficiency_mean efficiency_std time_mean time_std\n");
    printf("[%d]>>>> starpudepsstfmpi %d %d %d %d %d %d %d %e %d %e %e %e %e\n", my_rank, my_rank, nranks, repeat, n_threads, n_rows, n_edges, n_cols, spin_time, n_tasks, eff_mean, eff_std, time_mean, time_std);

    return 0;
}

int main(int argc, char **argv)
{
    int req = MPI_THREAD_SERIALIZED;
    int prov = -1;
    MPI_Init_thread(NULL, NULL, req, &prov);
    starpu_mpi_init_conf(&argc, &argv, 0, MPI_COMM_WORLD, NULL);

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

    starpu_mpi_shutdown();
    MPI_Finalize();
    return error;
}
