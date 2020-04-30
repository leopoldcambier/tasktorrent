#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <fstream>
#include <array>
#include <random>
#include <mutex>
#include <gtest/gtest.h>

#include "tasktorrent/tasktorrent.hpp"

using namespace std;
using namespace Eigen;
using namespace ttor;

typedef array<int, 2> int2;
typedef array<int, 3> int3;
typedef SparseMatrix<double> SpMat;

int VERB = 0;
int n_threads_ = 4;
int n_ = 50;
int N_ = 10;

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

PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> random_perm(int n)
{
    PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(n);
    perm.setIdentity();
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size(), g);
    return perm;
}

SpMat random_dag(int n, double p)
{
    SpMat A = random_SpMat(n, p, 2019).triangularView<StrictlyLower>();
    auto P = random_perm(n);
    return P.transpose() * A * P;
}

/**
 * Unit tests
 */

TEST(threadpool, tasksrun)
{
    int n_threads = n_threads_;
    int n_tasks = 10;
    vector<int> ok(n_tasks, 0);
    Threadpool tp(n_threads);

    for (int k = 0; k < n_tasks; k++)
    {
        Task *t = new Task();
        t->run = [k, &ok]() { ok[k] = 1; };
        t->fulfill = []() {};
        t->name = "Task_" + to_string(k);
        tp.insert(t, k % n_threads);
    }

    tp.join();

    for (int k = 0; k < n_tasks; k++)
        ASSERT_EQ(ok[k], 1);
}

/**
 * MiniApps
 */

TEST(graph, mini)
{
    int n_threads = n_threads_;

    // 0 = start immediately ; 1 == start before the FF ; 2 == start after the FF
    int start_kind[] = {0, 1, 2};
    for(auto start: start_kind) {

        Threadpool tp(n_threads, VERB, "WkTest_", (start == 0));
        Taskflow<int> mini_tf(&tp, VERB);
        std::atomic<int> ntasks_done(0);

        vector<vector<int>> out_deps = {
            vector<int>{1, 2},
            vector<int>{3},
            vector<int>{3, 5},
            vector<int>{4, 6},
            vector<int>{5},
            vector<int>{},
            vector<int>{},
            vector<int>{5}};

        vector<int> indegree = {1, 1, 1, 2, 1, 3, 1, 1};

        mini_tf.set_mapping([&](int k) {
                return (k % n_threads);
            })
            .set_indegree([&](int k) {
                return indegree[k];
            })
            .set_task([&](int k) {
                ntasks_done++;
                for (auto k_ : out_deps[k])
                    mini_tf.fulfill_promise(k_);
            })
            .set_name([](int k) {
                return "miniTask_" + to_string(k);
            })
            .set_priority([](int k) {
                return k;
            });
        
        if(start == 0) { } // Nothing
        if(start == 1) { tp.start(); } // Start before the FF

        mini_tf.fulfill_promise(0);
        mini_tf.fulfill_promise(7);

        if(start == 2) { tp.start(); } // Start after the FF

        tp.join();
        EXPECT_EQ(ntasks_done.load(), 8);
    }
}

TEST(reduction, concurrent)
{
    int n_threads = n_threads_;

    int n_reds = 123;
    int reds_size = 97;

    vector<atomic<bool>> is_running(n_reds);

    for (int i = 0; i < n_reds; i++)
        is_running[i].store(false);

    Threadpool tp(n_threads, VERB);
    Taskflow<int2> reduction_tf(&tp, VERB);

    reduction_tf.set_mapping([&](int2 ij) {
            return ij[0] % n_threads;
        })
        .set_binding([&](int2) {
            return true;
        })
        .set_indegree([](int2) {
            return 1;
        })
        .set_task([&](int2 ij) {
            ASSERT_FALSE(is_running.at(ij[0]).load());
            is_running.at(ij[0]).store(true);
            ASSERT_TRUE(is_running.at(ij[0]).load());
            is_running.at(ij[0]).store(false);
            ASSERT_FALSE(is_running.at(ij[0]).load());
        });

    for (int i = 0; i < n_reds; i++)
        for (int j = 0; j < reds_size; j++)
            reduction_tf.fulfill_promise({i, j});

    tp.join();
}

TEST(reduction, reduc)
{
    int n_threads = n_threads_;
    int n_reds = 2;

    atomic<bool> is_running(false);
    atomic<int> sum(0); // where we do all the reductions

    Threadpool tp(n_threads, VERB);
    Taskflow<int> tf(&tp, VERB); // first & last task
    Taskflow<int> tr(&tp, VERB); // all tasks in the middle

    tf.set_mapping([&](int i) {
          return i % n_threads;
      })
        .set_indegree([&](int i) {
            return (i == 0 ? 1 : n_reds);
        })
        .set_task([&](int i) {
            if (i == 0)
            {
                assert(sum.load() == 0);
                sum++;
                for (int j = 0; j < n_reds; j++)
                {
                    tr.fulfill_promise(j);
                }
            }
            else
            {
                assert(sum.load() == n_reds + 1);
                sum++;
            }
        })
        .set_name([&](int i) {
            return "TF_" + to_string(i);
        });

    tr.set_mapping([&](int) {
          return 0;
      })
        .set_binding([&](int) {
            return true;
        })
        .set_indegree([](int) {
            return 1;
        })
        .set_task([&](int) {
            assert(!is_running.load());
            is_running.store(true);
            assert(is_running.load());
            sum++;
            tf.fulfill_promise(1);
            is_running.store(false);
            assert(!is_running.load());
        })
        .set_name([&](int i) {
            return "TR_" + to_string(i);
        });

    tf.fulfill_promise(0);

    tp.join();

    EXPECT_EQ(sum, n_reds + 2);
}

/**
 * Generates a random DAG, check it runs
 */
TEST(graph, randomdag)
{
    for (auto n_threads : {1, 2, 4, 8, 16, 32})
    {
        for (auto n : {1, 3, 5, 7, 9, 15, 20, 30, 96, 200}) // # of tasks
        {
            for (auto p : {0.0, 0.001, 0.1, 0.2, 0.5}) // Fraction of edges out of the nxns are there
            {
                SpMat G = random_dag(n, p);

                // Count in_deps
                vector<int> indegree(n);
                for (int k = 0; k < G.outerSize(); ++k)
                {
                    for (SpMat::InnerIterator it(G, k); it; ++it)
                    {
                        indegree[it.row()]++;
                    }
                }

                vector<atomic<int>> tasks_ran(n);
                vector<atomic<int>> count_in_deps(n);

                for (int k = 0; k < n; k++)
                {
                    tasks_ran[k].store(0);
                    count_in_deps[k].store(indegree[k]);
                }

                Threadpool tp(n_threads, VERB);
                Taskflow<int> rand_graph_tf(&tp, VERB);

                rand_graph_tf.set_mapping([n_threads](int k) {
                                 return (k % n_threads);
                             })
                    .set_indegree([&indegree](int k) {
                        return max(1, indegree[k]);
                    })
                    .set_task([&G, &rand_graph_tf, &tasks_ran, &count_in_deps](int k) {
                        EXPECT_TRUE(count_in_deps[k].load() >= 0);
                        EXPECT_EQ(tasks_ran[k], 0);
                        tasks_ran[k]++;
                        for (SpMat::InnerIterator it(G, k); it; ++it)
                        {
                            rand_graph_tf.fulfill_promise(it.row());
                            count_in_deps[it.row()]--;
                            EXPECT_TRUE(count_in_deps[it.row()] >= 0);
                        }
                    });

                for (int i = 0; (size_t)i < indegree.size(); i++)
                    if (indegree[i] == 0)
                        rand_graph_tf.fulfill_promise(i);

                tp.join();

                for (int i = 0; i < n; i++)
                {
                    EXPECT_TRUE(tasks_ran[i].load() == 1);
                    EXPECT_TRUE(count_in_deps[i].load() == 0);
                }
            }
        }
    }
}

// Matrix-matrix product
void gemm(int n_threads, int n, int N)
{
    std::default_random_engine gen;
    // std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::uniform_int_distribution<> dist(-4, 4);
    auto rnd = [&](int, int) { return dist(gen); };

    MatrixXd A = MatrixXd::NullaryExpr(N * n, N * n, rnd);
    MatrixXd B = MatrixXd::NullaryExpr(N * n, N * n, rnd);
    MatrixXd C = MatrixXd::Zero(N * n, N * n);
    timer tref = wctime();
    MatrixXd Cref = A * B;
    timer trefend = wctime();

    Threadpool tp(n_threads, VERB);
    Taskflow<int3> gemm_tf(&tp, VERB);

    gemm_tf.set_mapping([n_threads, N](int3 id) {
               return ((id[0] + N * id[1]) % n_threads);
           })
        .set_indegree([](int3) {
            return 1;
        })
        .set_task([&A, &B, &C, n, N, &gemm_tf](int3 id) {
            int i = id[0], j = id[1], k = id[2];
            C.block(i * n, j * n, n, n) += A.block(i * n, k * n, n, n) * B.block(k * n, j * n, n, n);
            if (k < N - 1)
                gemm_tf.fulfill_promise({i, j, k + 1});
        });

    timer t0 = wctime();

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            gemm_tf.fulfill_promise({i, j, 0});

    tp.join();
    timer t1 = wctime();

    double err = (Cref - C).norm() / Cref.norm();

    if (VERB > 0)
        printf("%d threads, matrix size %dx%d, %dx%d blocks\n", n_threads, n * N, n * N, n, n);
    if (VERB > 0)
        printf("TTOR gemm+dag:   %f seconds elapsed\n", elapsed(t0, t1));
    if (VERB > 0)
        printf("EIGEN plain gemm: %f seconds elapsed\n", elapsed(tref, trefend));
    if (VERB > 0)
        printf("Error |C-Cref|/|Cref| %e\n", err);

    ASSERT_EQ(err, 0.); // We use int only so roundoff errors should be 0
}

TEST(gemm, all)
{
    for (auto n_threads : {1, 2, 4, 8, 16})
    {
        for (auto n : {1, 5, 25})
        {
            for (auto N : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
            {
                gemm(n_threads, n, N);
            }
        }
    }
}

TEST(gemm, one)
{
    int n_threads = n_threads_;
    int n = n_;
    int N = N_;
    gemm(n_threads, n, N);
}

void cholesky(int n_threads, int n, int N)
{

    std::default_random_engine gen;
    // std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::uniform_int_distribution<> dist(-1, 1);
    auto rnd = [&](int i, int j) { return i == j ? 4 : dist(gen); };

    MatrixXd A_ = MatrixXd::NullaryExpr(N * n, N * n, rnd);

    MatrixXd A = A_ * A_.transpose();
    A = A.triangularView<Lower>();
    MatrixXd Aref = A;

    Threadpool tp(n_threads, VERB);
    Taskflow<int> potf_tf(&tp, VERB);
    Taskflow<int2> trsm_tf(&tp, VERB);
    Taskflow<int3> gemm_tf(&tp, VERB);

    DepsLogger dlog(1000000);
    Logger log(1000000);
    tp.set_logger(&log);

    potf_tf.set_mapping([&](int k) {
               return (k % n_threads);
           })
        .set_indegree([](int) {
            return 1;
        })
        .set_task([&](int k) {
            LLT<MatrixXd> llt;
            llt.compute(A.block(k * n, k * n, n, n).selfadjointView<Lower>());
            A.block(k * n, k * n, n, n) = llt.matrixL();
        })
        .set_fulfill([&](int k) {
            for (int i = k + 1; i < N; i++)
            {
                trsm_tf.fulfill_promise({k, i});
                dlog.add_event(make_unique<DepsEvent>(potf_tf.name(k), trsm_tf.name({k, i})));
            }
        })
        .set_name([](int k) {
            return "potf_" + to_string(k);
        })
        .set_priority([](int) {
            return 3.0;
        });

    trsm_tf.set_mapping([&](int2 ki) {
               return ((ki[0] + ki[1] * N) % n_threads);
           })
        .set_indegree([](int2 ki) {
            return (ki[0] == 0 ? 0 : 1) + 1;
        })
        .set_task([&](int2 ki) {
            int k = ki[0];
            int i = ki[1];
            auto L = A.block(k * n, k * n, n, n).triangularView<Lower>().transpose();
            A.block(i * n, k * n, n, n) = L.solve<OnTheRight>(A.block(i * n, k * n, n, n));
        })
        .set_fulfill([&](int2 ki) {
            int k = ki[0];
            int i = ki[1];
            for (int l = k + 1; l <= i; l++)
            {
                gemm_tf.fulfill_promise({k, i, l});
                dlog.add_event(make_unique<DepsEvent>(trsm_tf.name(ki), gemm_tf.name({k, i, l})));
            }
            for (int l = i + 1; l < N; l++)
            {
                gemm_tf.fulfill_promise({k, l, i});
                dlog.add_event(make_unique<DepsEvent>(trsm_tf.name(ki), gemm_tf.name({k, l, i})));
            }
        })
        .set_name([](int2 ki) {
            return "trsm_" + to_string(ki[0]) + "_" + to_string(ki[1]);
        })
        .set_priority([](int2) {
            return 2.0;
        });

    gemm_tf.set_mapping([&](int3 kij) {
               return ((kij[0] + kij[1] * N + kij[2] * N * N) % n_threads);
           })
        .set_indegree([](int3 kij) {
            int k = kij[0];
            int i = kij[1];
            int j = kij[2];
            return (k == 0 ? 0 : 1) + (i == j ? 1 : 2);
        })
        .set_task([&](int3 kij) {
            int k = kij[0];
            int i = kij[1];
            int j = kij[2];
            A.block(i * n, j * n, n, n) -= A.block(i * n, k * n, n, n) * A.block(j * n, k * n, n, n).transpose();
            if (i == j)
                A.block(i * n, j * n, n, n) = A.block(i * n, j * n, n, n).triangularView<Lower>();
            ASSERT_TRUE(k < N - 1);
        })
        .set_fulfill([&](int3 kij) {
            int k = kij[0];
            int i = kij[1];
            int j = kij[2];
            if (k + 1 == i && k + 1 == j)
            {
                potf_tf.fulfill_promise(k + 1);
                dlog.add_event(make_unique<DepsEvent>(gemm_tf.name(kij), potf_tf.name(k + 1)));
            }
            else if (k + 1 == j)
            {
                trsm_tf.fulfill_promise({k + 1, i});
                dlog.add_event(make_unique<DepsEvent>(gemm_tf.name(kij), trsm_tf.name({k + 1, i})));
            }
            else
            {
                gemm_tf.fulfill_promise({k + 1, i, j});
                dlog.add_event(make_unique<DepsEvent>(gemm_tf.name(kij), gemm_tf.name({k + 1, i, j})));
            }
        })
        .set_name([](int3 kij) {
            return "gemm_" + to_string(kij[0]) + "_" + to_string(kij[1]) + "_" + to_string(kij[2]);
        })
        .set_priority([](int3) {
            return 1.0;
        });

    timer t0 = wctime();
    potf_tf.fulfill_promise(0);

    tp.join();
    timer t1 = wctime();

    LLT<MatrixXd> llt;
    timer t2 = wctime();
    llt.compute(Aref.selfadjointView<Lower>());
    timer t3 = wctime();
    MatrixXd Lref = llt.matrixL();

    if (VERB > 0)
    {
        printf("TTOR=%e s.\n", elapsed(t0, t1));
        printf("EIGEN=%e s.\n", elapsed(t2, t3));
    }

    double err = (A - Lref).norm() / Lref.norm();
    if (VERB > 0)
        cout << "Error=" << err << endl;
    EXPECT_TRUE(err < 1e-13) << err << endl;

    std::ofstream logfile;
    logfile.open("cholesky.log");
    logfile << log;
    logfile.close();

    std::ofstream depsfile;
    depsfile.open("cholesky.dot");
    depsfile << dlog;
    depsfile.close();
}

TEST(cholesky, all)
{
    for (auto n_threads : {1, 2, 4, 8, 16})
    {
        for (auto n : {1, 5, 25})
        {
            for (auto N : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
            {
                cholesky(n_threads, n, N);
            }
        }
    }
}

TEST(cholesky, many)
{
    cholesky(8, 1, 8);
    cholesky(16, 1, 8);
    cholesky(32, 1, 8);

    // cholesky(100, 5, 5);
    // cholesky(1000, 5, 5);
    // cholesky(1000, 3, 30);
}

TEST(cholesky, one)
{
    int n = n_;
    int N = N_;
    int n_threads = n_threads_;
    cholesky(n_threads, n, N);
}

TEST(ttor, mapreduce)
{

    struct Data
    {
        mutex data_lock;
        vector<int> data;
    };

    int n_threads = n_threads_;
    int N = 10000;
    int n_buckets = 250;
    int reduce_deps = N / n_buckets;
    assert(N % n_buckets == 0);

    Threadpool tp(n_threads, VERB);
    Taskflow<int> mf(&tp, VERB);
    Taskflow<int> rf(&tp, VERB);
    vector<Data> reddata(n_buckets);

    vector<int> data(N);
    default_random_engine gen;
    uniform_int_distribution<> dist(0, 10);
    for (int i = 0; i < N; i++)
        data[i] = dist(gen);

    vector<int> results(n_buckets);

    mf.set_mapping([&](int k) {
          return (k % n_threads);
      })
        .set_indegree([&](int) {
            return 1;
        })
        .set_task([&](int k) {
            int bucket = k % n_buckets;
            int val = data[k];
            reddata[bucket].data_lock.lock();
            reddata[bucket].data.push_back(val);
            reddata[bucket].data_lock.unlock();
            rf.fulfill_promise(bucket);
        })
        .set_name([](int k) {
            return "Map_" + to_string(k);
        });

    rf.set_mapping([&](int k) {
          return (k % n_threads);
      })
        .set_indegree([&](int) {
            return reduce_deps;
        })
        .set_task([&](int k) {
            int result = 0;
            for (auto v : reddata[k].data)
            {
                result += v;
            }
            results[k] = result;
        })
        .set_name([](int k) {
            return "Reduce_" + to_string(k);
        });

    for (int i = 0; i < N; i++)
        mf.fulfill_promise(i);

    tp.join();

    for (int i = 0; i < n_buckets; i++)
    {
        int expected = 0;
        for (int j = 0; j < reduce_deps; j++)
        {
            expected += data[i + j * n_buckets];
        }
        ASSERT_EQ(expected, results[i]);
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    if (argc >= 2)
    {
        n_threads_ = atoi(argv[1]);
        cout << "Number of threads set to " << n_threads_ << "\n";
    }

    if (argc >= 3)
    {
        n_ = atoi(argv[2]);
        cout << "Block size set to " << n_ << "\n";
    }

    if (argc >= 4)
    {
        N_ = atoi(argv[3]);
        cout << "Number of blocks set to " << N_ << "\n";
    }

    if (argc >= 5)
    {
        VERB = atoi(argv[4]);
        cout << "Verbosity level set to " << VERB << "\n";
    }

    const int return_flag = RUN_ALL_TESTS();

    return return_flag;
}
