#ifndef __TTOR_TESTS_COMMON__
#define __TTOR_TESTS_COMMON__

#include <chrono>
#include <numeric>
#include <cmath>
#include <vector>
#include <array>
#include <functional>

typedef std::array<int,2> int2;

std::chrono::time_point<std::chrono::high_resolution_clock> wtime_now() {
    return std::chrono::high_resolution_clock::now();
};

double wtime_elapsed(const std::chrono::time_point<std::chrono::high_resolution_clock>& t0, const std::chrono::time_point<std::chrono::high_resolution_clock>& t1) {
    return std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
};

void spin_for_seconds(double time) {
    auto t0 = std::chrono::high_resolution_clock::now();
    while(true) {
        auto t1 = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
        if( elapsed_time >= time ) break;
    }
}

void compute_stats(const std::vector<double> &data, double *average, double* std) {
    double mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
    double stddev = 0.0;
    for(int k = 0; k < data.size(); k++) {
        stddev += (data[k] - mean) * (data[k] - mean);
    }
    stddev = std::sqrt(stddev / (data.size() - 1.0));
    
    *average = mean;
    *std = stddev;
}

int get_starpu_num_cores() {
    const char* env_n_cores = std::getenv("STARPU_NCPU");
    if(env_n_cores == nullptr) { 
        printf("Missing STARPU_NCPU\n"); 
        exit(1); 
    }
    return atoi(env_n_cores);
}

void wait_only_run_repeat(
    const std::string experiment, 
    const int n_threads, 
    const int n_tasks, 
    const double spin_time, 
    const int repeat, 
    const int verb,
    std::function<double()> run
    ) 
{
    
    std::vector<double> efficiencies;
    std::vector<double> times;

    for(int step = 0; step < repeat; step++) {

        // Run once
        const double time = run();

        if(verb) printf("iteration repeat n_threads n_tasks spin_time time efficiency\n");
        const double speedup = (double)(n_tasks) * (double)(spin_time) / (double)(time);
        const double efficiency = speedup / (double)(n_threads);
        times.push_back(time);
        efficiencies.push_back(efficiency);
        printf("++++ %s %d %d %d %d %e %e %e\n", experiment.c_str(), step, repeat, n_threads, n_tasks, spin_time, time, efficiency);

    }

    double eff_mean, eff_std, time_mean, time_std;
    compute_stats(efficiencies, &eff_mean, &eff_std);
    compute_stats(times, &time_mean, &time_std);
    printf("]]]] exp repeat n_threads spin_time n_tasks efficiency_mean efficiency_std time_mean time_std\n");
    printf(">>>> %s %d %d %e %d %e %e %e %e\n", experiment.c_str(), repeat, n_threads, spin_time, n_tasks, eff_mean, eff_std, time_mean, time_std);

}

void deps_run_repeat(
    const std::string experiment, 
    const int n_threads, 
    const int n_rows, 
    const int n_edges, 
    const int n_cols, 
    const double spin_time, 
    const int repeat, 
    const int verb,
    std::function<double()> run
    ) 
{
    std::vector<double> efficiencies;
    std::vector<double> times;
    int n_tasks = n_rows * n_cols;

    for(int step = 0; step < repeat; step++) {

        // Run once
        double time = run();

        if(verb) printf("iteration repeat n_threads n_rows n_edges n_cols spin_time time n_tasks efficiency\n");
        double speedup = (double)(n_tasks) * (double)(spin_time) / (double)(time);
        double efficiency = speedup / (double)(n_threads);
        efficiencies.push_back(efficiency);
        times.push_back(time);
        printf("++++ %s %d %d %d %d %d %d %e %e %d %e\n", experiment.c_str(), step, repeat, n_threads, n_rows, n_edges, n_cols, spin_time, time, n_tasks, efficiency);

    }

    double eff_mean, eff_std, time_mean, time_std;
    compute_stats(efficiencies, &eff_mean, &eff_std);
    compute_stats(times, &time_mean, &time_std);
    printf("]]]] exp repeat n_threads n_rows n_edges n_cols spin_time n_tasks efficiency_mean efficiency_std time_mean time_std\n");
    printf(">>>> %s %d %d %d %d %d %e %d %e %e %e %e\n", experiment.c_str(), repeat, n_threads, n_rows, n_edges, n_cols, spin_time, n_tasks, eff_mean, eff_std, time_mean, time_std);
}

#endif
