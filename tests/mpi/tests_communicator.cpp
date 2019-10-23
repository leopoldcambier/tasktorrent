// make tests_comms_dag && mpirun -oversubscribe -n 8 ./tests_comms_dag 4 1 --gtest_repeat=16  --gtest_break_on_failure
// mpirun -mca btl ^tcp -oversubscribe -n 2 ./tests_comms_dag 2 2 --gtest_filter=*pingpong --gtest_repeat=8 | grep "Msg count update: sent - rcvd ="
// mpirun -mca shmem posix -mca btl ^tcp -oversubscribe
// -mca shmem posix: to remove error message when using oversubscribe
// -mca btl ^tcp: to remove firewall warnings on macos

#include "communications.hpp"
#include "runtime.hpp"
#include "util.hpp"

#include <fstream>
#include <array>
#include <random>
#include <exception>
#include <iostream>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <queue>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

using namespace std;
using namespace ttor;

int n_threads_ = 4;
int VERB = 0;

void trigger_task(Taskflow<int> &local_tf, int &dest)
{
    local_tf.fulfill_promise(dest);
}

struct local_t
{
    Taskflow<int> *local_tf;
};

TEST(ttor, mini_v0)
{
    int n_threads = n_threads_;
    int rank = comm_rank();

    Logger log(1);

    Communicator comm(VERB);

    Threadpool tp(n_threads, &comm, VERB, "WkMini_v0_" + to_string(rank) + "_");
    Taskflow<int> tf(&tp, VERB);

    vector<vector<int>> out_deps(2);
    vector<int> indegree(2);

    tp.set_logger(&log);
    comm.set_logger(&log);

    assert(indegree.size() == out_deps.size());
    assert(indegree.size() == 2);

    vector<int> task_completed(indegree.size(), 0);

    out_deps[0] = {1};
    out_deps[1] = {};

    indegree[0] = 1;
    indegree[1] = 1;

    tf.set_mapping([&](int k) {
          return (k % n_threads);
      })
        .set_indegree([&](int k) {
            return indegree[k];
        })
        .set_task([&](int k) {
            for (int k_ : out_deps[k])
                tf.fulfill_promise(k_);

            ASSERT_EQ(task_completed[k], 0);
            task_completed[k] = 1;
        })
        .set_name([&](int k) {
            return "mini_v0_Task_" + to_string(k) + "_" + to_string(rank);
        });

    tf.fulfill_promise(0);

    tp.join();

    for (int t = 0; t < task_completed.size(); ++t)
        ASSERT_EQ(task_completed[t], 1);

    MPI_Barrier(MPI_COMM_WORLD); // This is required in case we call this function repeatedly
}

TEST(ttor, mini_v1)
{
    int n_threads = n_threads_;
    int rank = comm_rank();

    Logger log(1);

    Communicator comm(VERB);

    Threadpool tp(n_threads, &comm, VERB, "WkMini_v1_" + to_string(rank) + "_");
    Taskflow<int> tf(&tp, VERB);

    tp.set_logger(&log);
    comm.set_logger(&log);

    vector<int> task_completed(n_threads, 0);

    tf.set_mapping([&](int k) {
          assert(k >= 0 && k < n_threads);
          return k;
      })
        .set_indegree([&](int k) {
            return 1;
        })
        .set_task([&](int k) {
            ASSERT_EQ(task_completed[k], 0);
            task_completed[k] = 1;
        })
        .set_name([&](int k) {
            return "mini_v1_Task_" + to_string(k) + "_" + to_string(rank);
        });

    for (int t = 0; t < task_completed.size(); ++t)
        tf.fulfill_promise(t);

    tp.join();

    for (int t = 0; t < task_completed.size(); ++t)
        ASSERT_EQ(task_completed[t], 1);

    MPI_Barrier(MPI_COMM_WORLD); // This is required in case we call this function repeatedly
}

template <class taskflow_t>
void test_ttor_mini(int n_tasks_per_rank, bool self)
{
    ASSERT_GE(n_tasks_per_rank, 0);

    int n_threads = n_threads_;
    int rank = comm_rank();
    int n_ranks = comm_size();

    if (VERB > 1)
        cout << "Hello from " << processor_name() << endl;

    auto task_2_rank = [&](int k) {
        assert(k >= 0 && k <= 7);
        const int rank = k / n_tasks_per_rank;
        assert(rank < n_ranks);
        return rank;
    };

    ASSERT_GT(n_ranks, task_2_rank(7));

    Logger log(1000000);

    Communicator comm(VERB);

    Threadpool tp(n_threads, &comm, VERB, "WkMini_" + to_string(rank) + "_");
    taskflow_t tf(&tp, VERB);

    auto am = comm.make_active_msg(
        [&](int &k_) {
            tf.fulfill_promise(k_);
        });

    unordered_map<int, vector<int>> out_deps;
    unordered_map<int, int> indegree;

    tp.set_logger(&log);
    comm.set_logger(&log);

    vector<int> task_completed(8, 0);

    out_deps[0] = {1, 2};
    out_deps[1] = {3};
    out_deps[2] = {3, 5};
    out_deps[3] = {4, 6};
    out_deps[4] = {5};
    out_deps[5] = {};
    out_deps[6] = {};
    out_deps[7] = {5};

    indegree[0] = 1;
    indegree[1] = 1;
    indegree[2] = 1;
    indegree[3] = 2;
    indegree[4] = 1;
    indegree[5] = 3;
    indegree[6] = 1;
    indegree[7] = 1;

    tf.set_mapping([&](int k) {
          return (k % n_threads);
      })
        .set_indegree([&](int k) {
            return indegree[k];
        })
        .set_task([&](int k) {
            for (int k_ : out_deps[k])
            {
                int dest = task_2_rank(k_);
                if (dest == rank && !self)
                    tf.fulfill_promise(k_);
                else
                    am->send(dest, k_);
            }
            ASSERT_EQ(task_completed[k], 0);
            task_completed[k] = 1;
        })
        .set_name([&](int k) {
            return "miniTask_" + to_string(k) + "_" + to_string(rank);
        });

    if (rank == task_2_rank(0))
        tf.fulfill_promise(0);

    if (rank == task_2_rank(7))
        tf.fulfill_promise(7);

    tp.join();

    for (int t = 0; t < 8; ++t)
    {
        if (rank == task_2_rank(t))
            ASSERT_EQ(task_completed[t], 1);
        else
            ASSERT_EQ(task_completed[t], 0);
    }

    if (VERB > 2)
    {
        std::ofstream logfile;
        logfile.open("commdag.log." + to_string(comm_rank()));
        logfile << log;
        logfile.close();
    }

    MPI_Barrier(MPI_COMM_WORLD); // This is required in case we call this function repeatedly
}

TEST(ttor, mini)
{
    const int n_ranks = comm_size();

    int n_tasks_per_rank = 8;
    while (n_tasks_per_rank > 0 && 7 / n_tasks_per_rank < n_ranks)
    {
        // We have to make sure that we have enough MPI ranks to process the DAG
        test_ttor_mini<Taskflow<int>>(n_tasks_per_rank, false);
        test_ttor_mini<Taskflow<int>>(n_tasks_per_rank, true);
        --n_tasks_per_rank;
    }
}

TEST(ttor, critical1)
{
    int n_threads = n_threads_;
    int rank = comm_rank();

    Logger log(1000000);

    Communicator comm(VERB);

    Threadpool tp(n_threads, &comm, VERB, "WkCritical_1_" + to_string(rank) + "_");
    Taskflow<int> tf_0(&tp, VERB);
    Taskflow<int> tf_1(&tp, VERB);

    tp.set_logger(&log);
    comm.set_logger(&log);

    int n_rounds = 10000;
    int pool_size = 1;
    vector<int> count_rounds(pool_size, 0);

    auto map = [pool_size](int k) { return k % pool_size; };

    tf_0.set_mapping([&](int k) {
            assert(k >= 0);
            return map(k);
        })
        .set_binding([&](int k) {
            return true;
        })
        .set_indegree([&](int k) {
            return 1;
        })
        .set_task([&](int k) {
            if (k < n_rounds * pool_size)
            {
                ++count_rounds[map(k)];
                tf_1.fulfill_promise(k);
                tf_0.fulfill_promise(k + pool_size);
            }
        })
        .set_name([&](int k) {
            return "critical_tf0_" + to_string(k) + "_" + to_string(rank);
        });

    tf_1.set_mapping([&](int k) {
            assert(k >= 0);
            return map(k);
        })
        .set_binding([&](int k) {
            return true;
        })
        .set_indegree([&](int k) {
            return 1;
        })
        .set_task([&](int k) {
            ++count_rounds[map(k)];
        })
        .set_name([&](int k) {
            return "critical_tf1_" + to_string(k) + "_" + to_string(rank);
        });

    for (int t = 0; t < pool_size; ++t)
        tf_0.fulfill_promise(t);

    tp.join();

    for (int t = 0; t < pool_size; ++t)
        ASSERT_EQ(count_rounds[t], 2 * n_rounds);

    MPI_Barrier(MPI_COMM_WORLD); // This is required in case we call this function repeatedly
}

TEST(ttor, critical2)
{
    int n_threads = n_threads_;
    int rank = comm_rank();

    Logger log(1000000);

    Communicator comm(VERB);

    Threadpool tp(n_threads, &comm, VERB, "WkCritical_2_" + to_string(rank) + "_");
    Taskflow<int> tf_0(&tp, VERB);
    Taskflow<int> tf_1(&tp, VERB);

    tp.set_logger(&log);
    comm.set_logger(&log);

    int n_work = 10000;
    int atomic_count = 0;

    tf_0.set_mapping([&](int k) {
            assert(k >= 0);
            return k % n_threads;
        })
        .set_binding([&](int k) {
            return false;
        })
        .set_indegree([&](int k) {
            return 1;
        })
        .set_task([&](int k) {
            tf_1.fulfill_promise(k);
        })
        .set_name([&](int k) {
            return "critical_tf0_" + to_string(k) + "_" + to_string(rank);
        });

    tf_1.set_mapping([&](int k) {
            return 0;
        })
        .set_binding([&](int k) {
            return true;
        })
        .set_indegree([&](int k) {
            return 1;
        })
        .set_task([&](int k) {
            ++atomic_count;
        })
        .set_name([&](int k) {
            return "critical_tf1_" + to_string(k) + "_" + to_string(rank);
        });

    for (int t = 0; t < n_work; ++t)
        tf_0.fulfill_promise(t);

    tp.join();

    ASSERT_EQ(atomic_count, n_work);

    MPI_Barrier(MPI_COMM_WORLD); // This is required in case we call this function repeatedly
}

const int graph_size = 32;

void test_sparse_graph(int n_tasks_per_rank, bool self)
{
    ASSERT_GE(n_tasks_per_rank, 0);

    int n_threads = n_threads_;
    int rank = comm_rank();
    int n_ranks = comm_size();

    auto task_2_rank = [&](int k) {
        assert(k >= 0 && k < graph_size);
        const int rank = k / n_tasks_per_rank;
        assert(rank < n_ranks);
        return rank;
    };

    ASSERT_GT(n_ranks, task_2_rank(graph_size - 1));

    Logger log(1000000);

    Communicator comm(VERB);

    Threadpool tp(n_threads, &comm, VERB, "WkSparseDAG_" + to_string(rank) + "_");
    Taskflow<int> tf(&tp, VERB);

    tp.set_logger(&log);
    comm.set_logger(&log);

    vector<int> task_completed(graph_size, 0);

    const int col_step = 1 + graph_size / 32;
    auto cols_rand = bind(uniform_int_distribution<int>(1, col_step), mt19937(2019));
    // Random number in the interval [1, col_step]
    // This adds more randomness; the number of nnz per row is slightly non-constant

    int n_edges = 0;

    vector<vector<int>> out_deps(graph_size);
    vector<int> indegree(graph_size, 0);

    indegree[0] = 1;

    for (int i = 0; i < graph_size; ++i)
    {
        queue<int> deps;

        // Add columns randomly (although we always add i+1 so that the reach of 0 is the complete graph)
        int col = i + 1;
        while (col < graph_size)
        {
            deps.push(col);
            col += cols_rand();
        }

        out_deps[i].resize(deps.size());
        n_edges += deps.size();
        int j = 0;
        while (!deps.empty())
        {
            out_deps[i][j] = deps.front();
            if (j > 0)
            {
                ASSERT_GT(out_deps[i][j], out_deps[i][j - 1]);
            }
            ++indegree[deps.front()];
            deps.pop();
            ++j;
        }
    }

    for (int i = 1; i < graph_size; ++i)
        ASSERT_GE(indegree[i], 1);

    ASSERT_EQ(indegree[0], 1);
    ASSERT_EQ(indegree[1], 1);

    auto am = comm.make_active_msg(
        [&](int &k) {
            assert(k >= 0 && k < graph_size);
            tf.fulfill_promise(k);
        });

    tf.set_mapping([&](int k) {
          return k % n_threads;
      })
        .set_indegree([&](int k) {
            return indegree[k];
        })
        .set_task([&](int k) {
            assert(k >= 0 && k < task_completed.size());

            for (int k_ : out_deps[k])
            {
                ASSERT_GT(k_, k);
                ASSERT_GE(k_, 0);
                ASSERT_LT(k_, graph_size);

                int dest = task_2_rank(k_);
                if (dest == rank && !self)
                    tf.fulfill_promise(k_);
                else
                    am->send(dest, k_);
            }
            ASSERT_EQ(task_completed[k], 0);
            task_completed[k] = 1;
        })
        .set_name([&](int k) {
            return "sparseTask_" + to_string(k) + "_" + to_string(rank);
        });

    if (rank == task_2_rank(0))
        tf.fulfill_promise(0);

    tp.join();

    for (int t = 0; t < graph_size; ++t)
    {
        if (rank == task_2_rank(t))
            ASSERT_EQ(task_completed[t], 1);
        else
            ASSERT_EQ(task_completed[t], 0);
    }

    MPI_Barrier(MPI_COMM_WORLD); // This is required in case we call this function repeatedly
}

TEST(ttor, sparse_graph)
{
    const int n_ranks = comm_size();

    int n_tasks_per_rank = graph_size;

    const int tsk_step = 1 + graph_size / 16;

    while (n_tasks_per_rank > 0 && (graph_size - 1) / n_tasks_per_rank < n_ranks)
    {
        // We have to make sure that we have enough MPI ranks to process the DAG
        test_sparse_graph(n_tasks_per_rank, false);
        test_sparse_graph(n_tasks_per_rank, true);
        n_tasks_per_rank -= tsk_step;
    }
}

TEST(ttor, ring)
{
    const int n_ranks = comm_size();
    int rank = comm_rank();
    const int dest = (rank + 1) % n_ranks;
    const int n_rounds = 8;
    const int expected = n_rounds * (rank > 0 ? rank - 1 : n_ranks - 1);

    Communicator comm(VERB);

    Threadpool tp(2, &comm, VERB, "WkRing_" + to_string(rank) + "_");

    Taskflow<int> recvf(&tp, VERB);
    Taskflow<int> sendf(&tp, VERB);

    struct local_t
    {
        int payload;
        vector<bool> received;
        vector<bool> sent;
        local_t() : payload(0), received(n_rounds, false), sent(n_rounds, false) {}
    };

    local_t data;
    int rounds = 0;

    auto am = comm.make_active_msg(
        [&](int &round, int &payload) {
            ASSERT_LT(round, n_rounds);
            data.payload += payload;
            ASSERT_FALSE(data.received[round]);
            data.received[round] = true;

            recvf.fulfill_promise(round);
        });

    recvf.set_mapping([](int round) {
             return 0;
         })
        .set_indegree([&](int round) {
            return 1;
        })
        .set_task([&](int round) {
            ASSERT_LT(round, n_rounds);
            ASSERT_TRUE(data.received[round]);

            if (rank > 0 || rounds < n_rounds)
                sendf.fulfill_promise(rank > 0 ? round : round + 1);
        })
        .set_name([rank](int i) {
            return ("Recv_" + to_string(rank));
        });

    sendf.set_mapping([](int round) {
             return 0;
         })
        .set_indegree([rank](int round) {
            return 1;
        })
        .set_task([&](int round) {
            ASSERT_LT(round, n_rounds);
            ASSERT_FALSE(data.sent[round]);
            data.sent[round] = true;
            ++rounds;

            am->send(dest, round, rank);
        })
        .set_name([rank](int round) {
            return ("Send_" + to_string(rank));
        });

    if (rank == 0)
        sendf.fulfill_promise(0);

    tp.join();

    ASSERT_EQ(data.payload, expected);
    ASSERT_TRUE(std::accumulate(data.sent.begin(), data.sent.end(), true, std::logical_and<bool>()));
    ASSERT_TRUE(std::accumulate(data.received.begin(), data.received.end(), true, std::logical_and<bool>()));
    ASSERT_EQ(rounds, n_rounds);

    MPI_Barrier(MPI_COMM_WORLD); // This is required in case we call this function repeatedly
}

TEST(ttor, pingpong)
{
    const int n_threads = n_threads_;
    const int n_rounds = 8;
    const int n_ranks = comm_size();
    int rank = comm_rank();
    ASSERT_TRUE(n_ranks % 2 == 0);
    const int other = (rank % 2 == 0 ? rank + 1 : rank - 1);

    if (VERB > 0)
        cout << "Pingpong " << rank << " <-> " << other << endl;

    const int expected = n_rounds * (rank + other);
    const int expected_lpcs = n_rounds;

    Communicator comm(VERB);

    Threadpool tp(n_threads, &comm, VERB, "Wk_" + to_string(rank) + "_");
    Taskflow<int> ppf(&tp, VERB);

    int ball = 0;
    int n_lpcs = 0;

    auto am = comm.make_active_msg(
        [&](int &round, int &new_ball) {
            ball = new_ball;
            const int next_round = (rank % 2 == 0) ? round + 1 : round;
            ppf.fulfill_promise(next_round);
        });

    ppf.set_mapping([&](int round) {
           return (round % n_threads);
       })
        .set_indegree([&](int round) {
            return 1;
        })
        .set_task([&](int round) {
            if (round < n_rounds)
            {
                ++n_lpcs;
                /* Not exactly correct since this is an lpc but we are not doing
                    anything on the last round. */
                ball += rank;
                am->send(other, round, ball);
            }
        })
        .set_name([rank](int round) {
            return ("PingPong_" + to_string(rank) + "_" + to_string(round));
        });

    if (rank % 2 == 0)
        ppf.fulfill_promise(0);

    tp.join();

    ASSERT_EQ(ball, expected);
    ASSERT_EQ(n_lpcs, expected_lpcs);

    MPI_Barrier(MPI_COMM_WORLD); // This is required in case we call this function repeatedly
}

int main(int argc, char **argv)
{
    int req = MPI_THREAD_FUNNELED;
    int prov = -1;
    MPI_Init_thread(NULL, NULL, req, &prov);
    assert(prov == req);

    if (argc >= 2)
    {
        n_threads_ = atoi(argv[1]);
        cout << "Number of threads set to " << n_threads_ << "\n";
    }

    if (argc >= 3)
    {
        VERB = atoi(argv[2]);
        cout << "Verbosity level set to " << VERB << "\n";
    }

    ::testing::InitGoogleTest(&argc, argv);
    const int return_flag = RUN_ALL_TESTS();
    assert(return_flag == 0);

    MPI_Finalize();
    return return_flag;
}
