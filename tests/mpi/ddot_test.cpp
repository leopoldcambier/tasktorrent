#include "communications.hpp"
#include "runtime.hpp"
#include "util.hpp"

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <fstream>
#include <array>
#include <random>
#include <mutex>
#include <iostream>
#include <map>

#include <gtest/gtest.h>
#include <mpi.h>

using namespace std;
using namespace Eigen;
using namespace ttor;

typedef array<int, 2> int2;
typedef array<int, 3> int3;

int VERB = 0;
int n_threads_ = 4;
int n_ = 50;

// Simple pseudo random number
// https://locklessinc.com/articles/prng/
typedef unsigned long long u64b;
int rng64(u64b *s)
{
    u64b c = 7319936632422683419ULL;
    u64b x = s[1];

    /* Increment 128bit counter */
    s[0] += c;
    s[1] += c + (s[0] < c);

    /* Two h iterations */
    x ^= x >> 32;
    x *= c;
    x ^= x >> 32;
    x *= c;

    /* Perturb result */
    return static_cast<int>(x + s[0]);
}

void ddot(int n_threads, int block_size)
{
    // MPI info
    const int rank = comm_rank();
    const int n_ranks = comm_size();

    // Problem data
    // Function used to initialize x and y
    // i is a global index into the array
    // x[i] and y[i] are initialized using the value returned by array_initializer(i)
    function<int(int)> array_initializer = [](int i) {
        u64b seed[2];
        seed[0] = i;
        seed[1] = i + 2019;
        return rng64(seed) % 1000; // Do not make the integer too large
    };

    VectorXd z = VectorXd::Zero(n_threads);

    double partial_sum = 0.0;
    double ddot_sum = 0.0;

    // Initialize the communicator structure
    Communicator comm(VERB);

    // Threadpool
    Threadpool tp(n_threads, &comm, VERB);

    Taskflow<int> dot_tf(&tp, VERB);
    Taskflow<int> add_tf(&tp, VERB);

    // Create active message
    auto am = comm.make_active_msg(
        [&ddot_sum](double &partial_sum) {
            ddot_sum += partial_sum;
        });

    // Log
    DepsLogger dlog(1000000);
    Logger log(1000000);
    tp.set_logger(&log);

    // task flow
    dot_tf.set_mapping([&](int k) {
              return (k % n_threads);
          })
        .set_name([&](int k) {
            return "ddot_" + to_string(k) + "_" + to_string(rank);
        })
        .set_indegree([](int) {
            return 1;
        })
        .set_task([&](int k) {
            const int global_index = rank * n_threads * block_size + k * block_size;
            auto a_init = [&, global_index](int i) {
                return array_initializer(global_index + i);
            };

            VectorXd x = VectorXd::NullaryExpr(block_size, a_init);
            VectorXd y = VectorXd::NullaryExpr(block_size, a_init);

            z[k] = x.dot(y);
        })
        .set_fulfill([&](int k) {
            add_tf.fulfill_promise(0); // same rank
            dlog.add_event(DepsEvent(dot_tf.name(k), add_tf.name(0)));
        });

    add_tf.set_mapping([&](int) {
              return 0;
          })
        .set_name([&](int k) {
            return "add_" + to_string(k) + "_" + to_string(rank);
        })
        .set_indegree([&](int) {
            return n_threads;
        })
        .set_task([&](int) {
            partial_sum = z.sum();
        })
        .set_fulfill([&](int k) {
            am->send(0, partial_sum);
            dlog.add_event(DepsEvent(add_tf.name(k), "Reduce"));
        });

    // Seed tasks
    for (int i = 0; i < n_threads; ++i)
        dot_tf.fulfill_promise(i);

    tp.join();

    if (rank == 0)
    {
        VectorXd x = VectorXd::NullaryExpr(n_ranks * n_threads * block_size, array_initializer);
        VectorXd y = VectorXd::NullaryExpr(n_ranks * n_threads * block_size, array_initializer);

        double ddot_ref = x.dot(y);
        auto err = abs(ddot_sum - ddot_ref);

        if (VERB > 0 && err != 0)
            printf("Error: %g, sum = %g; ref = %g\n", err, ddot_sum, ddot_ref);

        ASSERT_EQ(err, 0);
    }

    // Logging
    std::ofstream logfile;
    string filename = "ddot_" + to_string(rank) + ".log";
    logfile.open(filename);
    logfile << log;
    logfile.close();

    std::ofstream depsfile;
    string dfilename = "ddot_" + to_string(rank) + ".dot";
    depsfile.open(dfilename);
    depsfile << dlog;
    depsfile.close();
}

TEST(ddot, one)
{
    int n_threads = n_threads_;
    int n = n_;
    ddot(n_threads, n);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    int req = MPI_THREAD_FUNNELED;
    int prov = -1;

    MPI_Init_thread(NULL, NULL, req, &prov);

    assert(prov == req);

    if (argc >= 2)
    {
        n_threads_ = atoi(argv[1]);
    }

    if (argc >= 3)
    {
        n_ = atoi(argv[2]);
    }

    if (argc >= 4)
    {
        VERB = atoi(argv[3]);
    }

    if (VERB > 0)
        printf("# threads = %d; block size = %d; VERB = %d\n", n_threads_, n_, VERB);

    const int return_flag = RUN_ALL_TESTS();

    MPI_Finalize();

    return return_flag;
}
