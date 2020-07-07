#include <fstream>
#include <array>
#include <random>
#include <mutex>
#include <iostream>
#include <map>
#include <memory>

#include "tasktorrent/tasktorrent.hpp"

void spin_for(double time) {
    auto t0 = ttor::wctime();
    while(true) {
    	auto t1 = ttor::wctime();
        if( ttor::elapsed(t0, t1) >= time ) break;
    }
}

typedef std::array<int,2> int2;
typedef std::array<int,3> int3;

int cholesky(const int n_threads, const int n_blocks, const double sleep_for, const int verb)
{
    // Map threads to ranks
    auto block_2_thread = [&](int i, int j) {
        return (i + j * n_blocks) % n_threads;
    };

    // Initialize the runtime structures
    ttor::Threadpool_shared tp(n_threads, 0, "Wk_Chol_", false);
    ttor::Taskflow<int>  potrf(&tp, 0);
    ttor::Taskflow<int2> trsm(&tp, 0);
    ttor::Taskflow<int3> gemm(&tp, 0);
    std::atomic<size_t> n_tasks(0);

    potrf.set_task([&](int j) {
            spin_for(sleep_for);
            n_tasks++;
        })
        .set_fulfill([&](int j) {
            for (int i = j+1; i < n_blocks; i++) {
                trsm.fulfill_promise({i,j});
            }
        })
        .set_indegree([&](int j) {
            return 1;
        })
        .set_mapping([&](int j) {
            return block_2_thread(j, j);
        });

    trsm.set_task([&](int2 ij) {
            spin_for(sleep_for);
            n_tasks++;
        })
        .set_fulfill([&](int2 ij) {
            int i = ij[0];
            int j = ij[1];
            assert(i > j);
            for (int k = j+1; k < n_blocks; k++) {
                int ii = std::max(i,k);
                int jj = std::min(i,k);
                gemm.fulfill_promise({j,ii,jj});
            }
        })
        .set_indegree([&](int2 ij) {
            return 1 + (ij[1] == 0 ? 0 : 1); // Potrf and last gemm before
        })
        .set_mapping([&](int2 ij) {
            return block_2_thread(ij[0], ij[1]);
        });

    gemm.set_task([&](int3 kij) {
            spin_for(sleep_for);
            n_tasks++;
        })
        .set_fulfill([&](int3 kij) {
            const int k = kij[0];
            const int i = kij[1];
            const int j = kij[2];
            if (k < j-1) {
                gemm.fulfill_promise({k+1, i, j});
            } else {
                if (i == j) {
                    potrf.fulfill_promise(i);
                } else {
                    trsm.fulfill_promise({i,j});
                }
            }
        })
        .set_indegree([&](int3 kij) {
            return (kij[1] == kij[2] ? 1 : 2) + (kij[0] == 0 ? 0 : 1); // one potrf or two trsms + the gemm before
        })
        .set_mapping([&](int3 kij) {
            return block_2_thread(kij[1], kij[2]); // IMPORTANT if accumulate_parallel is true
        });

    potrf.fulfill_promise(0);
    auto t0 = ttor::wctime();
    tp.start();
    tp.join();
    auto t1 = ttor::wctime();
    double time = ttor::elapsed(t0, t1);
    if(verb) printf("n_threads,n_blocks,sleep_time,time,total_tasks,efficiency\n");
    int total_tasks = n_tasks.load();
    double speedup = (double)(total_tasks) * (double)(sleep_for) / (double)(time);
    double efficiency = speedup / (double)(n_threads);
    printf("%d,%d,%e,%e,%d,%e\n", n_threads, n_blocks, sleep_for, time, total_tasks, efficiency);
    return 0;
}

int main(int argc, char **argv)
{
    int n_threads = 1;
    int n_blocks = 10;
    double sleep_for = 1e-6;
    int verb = 0;

    if (argc >= 2)
    {
        n_threads = atoi(argv[1]);
        assert(n_threads > 0);
    }
    if (argc >= 3)
    {
        n_blocks = atoi(argv[2]);
        assert(n_blocks >= 0);
    }
    if (argc >= 4)
    {
        sleep_for = atof(argv[3]);
        assert(sleep_for >= 0);
    }
    if (argc >= 5)
    {
        verb = atoi(argv[4]);
        assert(verb >= 0);
    }

    if(verb) printf("./micro_cholesky n_threads n_blocks sleep_for verb\n");
    int error = cholesky(n_threads, n_blocks, sleep_for, verb);

    return error;
}
