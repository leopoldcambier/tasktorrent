#include <random>
#include <map>
#include <tuple>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <gtest/gtest.h>

#include "tasktorrent/tasktorrent.hpp"

using namespace std;
using namespace ttor;
using namespace Eigen;

typedef SparseMatrix<double> SpMat;

int VERB = 0;

SpMat random_SpMat(int n, double p, int seed)
{
    default_random_engine gen;
    gen.seed(seed);
    uniform_real_distribution<double> dist(0.0, 1.0);
    vector<Triplet<double>> triplets;
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            auto v_ij = dist(gen);
            if (v_ij < p)
            {
                triplets.push_back(Triplet<double>(i, j, v_ij));
            }
        }
    }
    SpMat A(n, n);
    A.setFromTriplets(triplets.begin(), triplets.end());
    return A;
}

PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> random_perm(int n, int seed)
{
    PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(n);
    perm.setIdentity();
    default_random_engine gen;
    gen.seed(seed);
    std::shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size(), gen);
    return perm;
}

SpMat random_dag(int n, double p, int seed)
{
    SpMat A = random_SpMat(n, p, seed).triangularView<StrictlyLower>();
    auto P = random_perm(n, seed);
    return P.transpose() * A * P;
}

void test(int n_threads, int n, double p) {

    // Create a random dag
    SpMat G = random_dag(n, p, n * p + 2020);

    const int rank = comms_world_rank();
    const int n_ranks = comms_world_size();
    default_random_engine gen;
    gen.seed(n * p + 2021);

    // Assign tasks to ranks
    vector<int> task_2_rank(n, 0);
    std::uniform_int_distribution<> dist1(0, n_ranks-1);
    for(int i = 0; i < n; i++) task_2_rank[i] = dist1(gen);
 
    // Decide if header or header+body
    vector<int> msg_kind(n, 0);
    std::uniform_int_distribution<> dist2(0, 1);
    for(int i = 0; i < n; i++) msg_kind[i] = dist2(gen);

    // Count in_deps
    vector<int> in_degree(n, 0);
    for (int k = 0; k < G.outerSize(); ++k) {
        for (SpMat::InnerIterator it(G, k); it; ++it) {
            in_degree[it.row()]++;
        }
    }

    auto comm = make_communicator_world(VERB);
    Threadpool tp(n_threads, comm.get(), VERB);
    Taskflow<int> tf(&tp, VERB);

    vector<int> buff_recv(n, 0);
    vector<int> task_ran(n, 0);
    vector<int> large_order_check(n * n, 0);

    auto am_large = comm->make_large_active_msg(
                [&](const int& source, const int& dest) {
                    EXPECT_NE(task_2_rank.at(source), rank);
                    EXPECT_EQ(task_2_rank.at(dest), rank);
                    EXPECT_EQ(large_order_check.at(source * n + dest), 1);
                    tf.fulfill_promise(dest);
                },
                [&](const int& source, const int& dest) {
                    EXPECT_NE(task_2_rank.at(source), rank);
                    EXPECT_EQ(task_2_rank.at(dest), rank);
                    EXPECT_EQ(large_order_check.at(source * n + dest), 0);
                    large_order_check[source * n + dest] = 1;
                    return &buff_recv[source]; // We're receiving 0 elements
                },
                [&](const int& source, const int& dest){
                    EXPECT_EQ(task_2_rank.at(source), rank);
                    EXPECT_NE(task_2_rank.at(dest), rank);
                });

    auto am = comm->make_active_msg(
                [&](const int& source, const int& dest) {
                    EXPECT_NE(task_2_rank.at(source), rank);
                    EXPECT_EQ(task_2_rank.at(dest), rank);
                    tf.fulfill_promise(dest);
                });

    tf.set_mapping([&](int k) {
        EXPECT_EQ(task_2_rank.at(k), rank);
        return (k % n_threads);
    })
    .set_indegree([&](int k) {
        EXPECT_EQ(task_2_rank.at(k), rank);
        return std::max(1, in_degree[k]);
    })
    .set_task([&](int k) {
        EXPECT_EQ(task_2_rank.at(k), rank);
        EXPECT_EQ(task_ran[k], 0);
        task_ran[k] ++;
        for (SpMat::InnerIterator it(G, k); it; ++it)
        {
            int other = it.row();
            int dest = task_2_rank.at(other);
            if(dest == rank) {
                tf.fulfill_promise(other);
            } else {
                if(msg_kind.at(k) == 0) {
                    am->send(dest, k, other);
                } else {
                    auto v = view<int>();
                    am_large->send_large(dest, v, k, other);
                }
            }
        }
    });

    for (int i = 0; i < n; i++) {
        if(in_degree[i] == 0 && task_2_rank.at(i) == rank) {
            tf.fulfill_promise(i);
        }
    }

    tp.join();

    comms_world_barrier();

    for (int i = 0; i < n; i++) {
        if(task_2_rank.at(i) == rank) {
            EXPECT_EQ(task_ran[i], 1);
        }
    }
}

class DagProblemTest : public ::testing::Test, public ::testing::WithParamInterface<tuple<int, int, double>> {};

TEST_P(DagProblemTest, MixedTwoSteps) {
    int n_threads;
    int n;
    double p;
    std::tie(n_threads, n, p) = GetParam();
    test(n_threads, n, p);
}

INSTANTIATE_TEST_SUITE_P(
    DagProblem, DagProblemTest,
    ::testing::Combine(
        ::testing::Values(1, 2, 4),
        ::testing::Values(1, 3, 5, 7, 9, 15, 20, 100, 1000),
        ::testing::Values(0.0, 0.001, 0.01, 0.1, 0.2)
    )
);

int main(int argc, char **argv)
{
    comms_init();

    ::testing::InitGoogleTest(&argc, argv);

    if (argc >= 2)
    {
        VERB = atoi(argv[1]);
    }

    if (VERB > 0)
        printf("VERB = %d\n", VERB);

    const int return_flag = RUN_ALL_TESTS();

    comms_finalize();

    return return_flag;
}
