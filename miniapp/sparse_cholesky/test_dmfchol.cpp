#include <fstream>
#include <array>
#include <random>
#include <fstream>
#include <iostream>
#include <set>
#include <array>
#include <random>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <exception>
#include <map>
#include <mutex>
#include <tuple>
#include <memory>

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>

#include <scotch.h>
#include <gtest/gtest.h>
#include <upcxx/upcxx.hpp>

#ifdef USE_MKL
#include <mkl_cblas.h>
#include <mkl_lapacke.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif

#include "runtime.hpp"
#include "util.hpp"
#include "mmio.hpp"

using namespace std;
using namespace Eigen;
using namespace ttor;

typedef Eigen::SparseMatrix<double> SpMat;

int VERB = 0;
int N_THREADS = 4;
string FILENAME = "neglapl_2_128.mm";
int N_LEVELS = 10;

vector<int> get_subids(vector<int> &p_rows, vector<int> &c_rows, int cs, int cn)
{
    assert(p_rows.size() >= cn);
    assert(cs + cn == c_rows.size());
    vector<int> subids(cn);
    int l = 0;
    for (int i = 0; i < cn; i++)
    {
        while (p_rows[l] != c_rows[cs + i])
        {
            l++;
        }
        subids[i] = l;
    }
    return subids;
};

typedef shared_ptr<vector<int>> pvi;
typedef shared_ptr<vector<double>> pvd;

struct Front
{
    int id;
    int start;
    int self_size;
    int nbr_size;
    pvi rows;
    pvd front;
    pvd xsol;
    vector<int> cid;
    int pid;

    // Temporaries for factor, extend-add, fwd and bwd solve
    vector<int> children_s;
    vector<int> children_n;
    vector<pvi> children_rows;
    vector<pvd> children_fronts;
    vector<pvd> children_xsol;
    pvd parent_xsol;
    pvi parent_rows;
    int fwdready;
    int bwdready;

    Front(int id_, int start_, int size_) : id(id_), start(start_), self_size(size_)
    {
        nbr_size = 0;
        pid = -1;
        rows = nullptr;
        front = nullptr;
        xsol = nullptr;
        parent_xsol = nullptr;
        parent_rows = nullptr;
        fwdready = -1;
        bwdready = -1;
    };
    void set_child_front(int id, int s, int n, pvi rows, pvd front)
    {
        auto pos = find(cid.begin(), cid.end(), id);
        assert(pos != cid.end());
        int i = distance(cid.begin(), pos);
        children_s[i] = s;
        children_n[i] = n;
        children_rows[i] = rows;
        children_fronts[i] = front;
    }
    void set_child_xsol(int id, int s, int n, pvi rows, pvd xsol)
    {
        auto pos = find(cid.begin(), cid.end(), id);
        assert(pos != cid.end());
        int i = distance(cid.begin(), pos);
        children_s[i] = s;
        children_n[i] = n;
        children_rows[i] = rows;
        children_xsol[i] = xsol;
    }
    void set_parent_xsol(int pid_, pvi rows, pvd xsol)
    {
        assert(pid == pid_);
        parent_rows = rows;
        parent_xsol = xsol;
    }
    void extend_add()
    {
        if (VERB > 1)
            printf("[%d] EA %d\n", upcxx::comm_rank(), id);
        for (int k = 0; k < cid.size(); k++)
        {
            auto cs = children_s[k];
            auto cn = children_n[k];
            auto c_rows = children_rows[k];
            auto c_front = children_fronts[k];
            assert(c_rows != nullptr);
            auto subids = get_subids(*rows, *c_rows, cs, cn);
            auto csize = cs + cn;
            auto thissize = self_size + nbr_size;
            assert(c_rows->size() == csize);
            assert(c_front->size() == csize * csize);
            assert(subids.size() == cn);
            for (int j = 0; j < cn; j++)
            {
                int fj = subids[j];
                for (int i = 0; i < cn; i++)
                {
                    int fi = subids[i];
                    (*front)[fi + fj * thissize] += (*c_front)[(cs + i) + (cs + j) * csize];
                }
            }
        }
    }
    void factor()
    {
        if (VERB > 1)
            printf("[%d] Factor %d\n", upcxx::comm_rank(), id);
        int lda = self_size + nbr_size;
        double *Ass = front->data();
        double *Ans = front->data() + self_size;
        double *Ann = front->data() + self_size * lda + self_size;
        // POTRF
        int err = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', self_size, Ass, lda);
        assert(err == 0);
        // TRSM
        cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit, nbr_size, self_size, 1.0, Ass, lda, Ans, lda);
        // SYRK
        cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans, nbr_size, self_size, -1.0, Ans, lda, 1.0, Ann, lda);
    };
    void fwd()
    {
        if (VERB > 1)
            printf("[%d] Fwd %d\n", upcxx::comm_rank(), id);
        // Add from children
        for (int k = 0; k < cid.size(); k++)
        {
            auto c_rows = children_rows[k];
            assert(c_rows != nullptr);
            auto c_xsol = children_xsol[k];
            auto cs = children_s[k];
            auto cn = children_n[k];
            auto subids = get_subids(*rows, *c_rows, cs, cn);
            for (int k = 0; k < cn; k++)
            {
                (*xsol)[subids[k]] += (*c_xsol)[cs + k];
            }
        }
        // Fwd solve
        double *Lss = front->data();
        double *Lns = front->data() + self_size;
        double *xs = xsol->data();
        double *xn = xsol->data() + self_size;
        int size = self_size + nbr_size;
        // x[s] <- Lss^-1 x[s]
        cblas_dtrsv(CblasColMajor, CblasLower, CblasNoTrans, CblasNonUnit, self_size, Lss, size, xs, 1);
        // xn = xn - Lns x[s]
        cblas_dgemv(CblasColMajor, CblasNoTrans, nbr_size, self_size, -1.0, Lns, size, xs, 1, 1.0, xn, 1);
    }
    void bwd()
    {
        // Set from parent
        if (pid != -1)
        {
            assert(parent_rows != nullptr);
            auto subids = get_subids(*parent_rows, *rows, self_size, nbr_size);
            for (int k = 0; k < nbr_size; k++)
            {
                (*xsol)[self_size + k] = (*parent_xsol)[subids[k]];
            }
        }
        // Bwd solve
        if (VERB > 1)
            printf("[%d] Bwd %d\n", upcxx::comm_rank(), id);
        double *Lss = front->data();
        double *Lns = front->data() + self_size;
        double *xs = xsol->data();
        double *xn = xsol->data() + self_size;
        int size = self_size + nbr_size;
        // x[s] -= Lns^T xn
        cblas_dgemv(CblasColMajor, CblasTrans, nbr_size, self_size, -1.0, Lns, size, xn, 1, 1.0, xs, 1);
        // xs = Lss^-T xs
        cblas_dtrsv(CblasColMajor, CblasLower, CblasTrans, CblasNonUnit, self_size, Lss, size, xs, 1);
    }
    void set_solution(VectorXd &xglob)
    {
        if (VERB > 1)
            printf("[%d] Set sol on %d\n", upcxx::comm_rank(), id);
        for (int i = 0; i < self_size; i++)
        {
            (*xsol)[i] = xglob[(*rows)[i]];
        }
    }
};

struct DistMF;
using dobj = upcxx::dist_object<DistMF *>;

struct DistMF
{
    map<int, shared_ptr<Front>> fronts;
    Taskflow<int> *fact_tasks_pt;
    Taskflow<int> *fwd_tasks_pt;
    Taskflow<int> *bwd_tasks_pt;

    VectorXi perm;
    SpMat A;
    int nblk;
    vector<int> fronts2ranks;

    int solve_not_done;
    dobj dmf;

    int front2rank(int id)
    {
        assert(id >= 0);
        int rank = fronts2ranks[id];
        assert(rank >= 0 && rank < upcxx::comm_size());
        return rank;
    }

    void compute_rank(shared_ptr<Front> f, int low, int high)
    {
        int nkids = f->cid.size();
        int mid = (low + high) / 2;
        fronts2ranks[f->id] = mid;
        for (int k = 0; k < nkids; k++)
        {
            int start = low + ((k) * (high - low)) / nkids;
            int end = low + ((k + 1) * (high - low)) / nkids;
            compute_rank(fronts[f->cid[k]], start, end);
        }
    }

    void send_front_to_parent(int id)
    {
        auto f = fronts.at(id);
        int dest = front2rank(f->pid);
        if (dest == upcxx::comm_rank())
        {
            auto fp = fronts.at(f->pid);
            fp->set_child_front(f->id, f->self_size, f->nbr_size, f->rows, f->front);
            fact_tasks_pt->fulfill_promise(f->pid);
        }
        else
        {
            if (VERB > 0)
                printf("Sending %d->%d, approx %lu bytes\n", upcxx::comm_rank(), dest, f->rows->size() * sizeof(int) + f->front->size() * sizeof(double));
            upcxx::rpc_ff(dest,
                          [](dobj &lmf, int id, int pid, int self_size, int nbr_size, upcxx::view<int> rows, upcxx::view<double> front) {
                              if (VERB > 0)
                                  printf("[%d] received %lu bytes\n", upcxx::comm_rank(), rows.size() * sizeof(int) + front.size() * sizeof(double));
                              auto f = (*lmf)->fronts.at(pid);
                              auto prows = make_shared<vector<int>>(rows.begin(), rows.end());
                              auto pfront = make_shared<vector<double>>(front.begin(), front.end());
                              f->set_child_front(id, self_size, nbr_size, prows, pfront);
                              (*lmf)->fact_tasks_pt->fulfill_promise(pid);
                          },
                          dmf, f->id, f->pid, f->self_size, f->nbr_size,
                          upcxx::make_view(f->rows->begin(), f->rows->end()),
                          upcxx::make_view(f->front->begin(), f->front->end()));
        }
    }

    void send_xsol_to_parent(int id)
    {
        auto f = fronts.at(id);
        int dest = front2rank(f->pid);
        if (dest == upcxx::comm_rank())
        {
            auto fp = fronts.at(f->pid);
            fp->set_child_xsol(f->id, f->self_size, f->nbr_size, f->rows, f->xsol);
            fwd_tasks_pt->fulfill_promise(f->pid);
        }
        else
        {
            rpc_ff(dest,
                   [](dobj &lmf, int pid, int id, int self_size, int nbr_size, upcxx::view<int> rows, upcxx::view<double> xsol) {
                       auto f = (*lmf)->fronts.at(pid);
                       auto prows = make_shared<vector<int>>(rows.begin(), rows.end());
                       auto pxsol = make_shared<vector<double>>(xsol.begin(), xsol.end());
                       f->set_child_xsol(id, self_size, nbr_size, prows, pxsol);
                       (*lmf)->fwd_tasks_pt->fulfill_promise(pid);
                   },
                   dmf, f->pid, f->id, f->self_size, f->nbr_size,
                   upcxx::make_view(f->rows->begin(), f->rows->end()),
                   upcxx::make_view(f->xsol->begin(), f->xsol->end()));
        }
    }

    void send_xsol_to_children(int id)
    {
        auto f = fronts.at(id);
        for (auto cid : f->cid)
        {
            int dest = front2rank(cid);
            if (dest == upcxx::comm_rank())
            {
                auto fc = fronts.at(cid);
                fc->set_parent_xsol(f->id, f->rows, f->xsol);
                bwd_tasks_pt->fulfill_promise(cid);
            }
            else
            {
                rpc_ff(dest,
                       [](dobj &lmf, int cid, int id, upcxx::view<int> rows, upcxx::view<double> xsol) {
                           auto f = (*lmf)->fronts.at(cid);
                           auto prows = make_shared<vector<int>>(rows.begin(), rows.end());
                           auto pxsol = make_shared<vector<double>>(xsol.begin(), xsol.end());
                           f->set_parent_xsol(id, prows, pxsol);
                           (*lmf)->bwd_tasks_pt->fulfill_promise(cid);
                       },
                       dmf, cid, f->id,
                       upcxx::make_view(f->rows->begin(), f->rows->end()),
                       upcxx::make_view(f->xsol->begin(), f->xsol->end()));
            }
        }
    }

    void fetch_xsol(int id, VectorXd &xglob)
    {
        int dest = front2rank(id);
        tuple<vector<double>, int> pair = rpc(dest, [](dobj &lmf, int fid) {
                                              auto f = (*lmf)->fronts.at(fid);
                                              vector<double> xs(f->xsol->begin(), f->xsol->begin() + f->self_size);
                                              (*lmf)->solve_not_done--;
                                              return make_tuple(xs, f->start);
                                          },
                                              dmf, id)
                                              .wait();
        vector<double> &xs = get<0>(pair);
        int start = get<1>(pair);
        for (int i = 0; i < xs.size(); i++)
        {
            xglob[start + i] = xs[i];
        }
    }

    DistMF(string filename, int nlevels) : nblk(-1), solve_not_done(-1), dmf(this)
    {
        if (upcxx::comm_rank() == 0)
            cout << "Matrix file " << filename << endl;
        // Read matrix
        A = mmio::sp_mmread<double, int>(filename);
        // Initialize & prepare
        int N = A.rows();
        int nnz = A.nonZeros();
        // Create rowval and colptr
        VectorXi rowval(nnz);
        VectorXi colptr(N + 1);
        int k = 0;
        colptr[0] = 0;
        for (int j = 0; j < N; j++)
        {
            for (SpMat::InnerIterator it(A, j); it; ++it)
            {
                int i = it.row();
                if (i != j)
                {
                    rowval[k] = i;
                    k++;
                }
            }
            colptr[j + 1] = k;
        }
        // Create SCOTCH graph
        SCOTCH_Graph *graph = SCOTCH_graphAlloc();
        int err = SCOTCH_graphInit(graph);
        assert(err == 0);
        err = SCOTCH_graphBuild(graph, 0, N, colptr.data(), nullptr, nullptr, nullptr, k, rowval.data(), nullptr);
        assert(err == 0);
        err = SCOTCH_graphCheck(graph);
        assert(err == 0);
        // Create strat
        SCOTCH_Strat *strat = SCOTCH_stratAlloc();
        err = SCOTCH_stratInit(strat);
        assert(err == 0);
        assert(nlevels > 0);
        string orderingstr = "n{sep=(/levl<" + to_string(nlevels - 1) + "?g:z;)}";
        err = SCOTCH_stratGraphOrder(strat, orderingstr.c_str());
        assert(err == 0);
        // Order with SCOTCH
        VectorXi permtab(N);
        VectorXi peritab(N);
        VectorXi rangtab(N + 1);
        VectorXi treetab(N);
        err = SCOTCH_graphOrder(graph, strat, permtab.data(), peritab.data(), &nblk, rangtab.data(), treetab.data());
        assert(err == 0);
        assert(nblk >= 0);
        if (VERB > 0)
            printf("[%d] Scotch ordering OK with %d blocs\n", upcxx::comm_rank(), nblk);
        treetab.conservativeResize(nblk);
        // Permute matrix
        SpMat App = permtab.asPermutation() * A * permtab.asPermutation().transpose();
        perm = permtab;
        // Create all the fronts, find childrens & parent
        for (int b = 0; b < nblk; b++)
        {
            int start = rangtab[b];
            ;
            int self_size = rangtab[b + 1] - rangtab[b];
            fronts[b] = shared_ptr<Front>(new Front(b, start, self_size));
        }
        for (int b = 0; b < nblk; b++)
        {
            int p = treetab[b];
            fronts[b]->pid = p;
            assert(p == -1 || p > b);
            if (p != -1)
            {
                fronts[p]->cid.push_back(b);
            }
        }
        // Decide on ranks
        fronts2ranks = vector<int>(nblk);
        for (int b = 0; b < nblk; b++)
        {
            if (fronts[b]->pid == -1)
            {
                compute_rank(fronts[b], 0, upcxx::comm_size());
            }
        }
        // Find neighbors
        for (int b = 0; b < nblk; b++)
        {
            auto f = fronts[b];
            int start = f->start;
            int self_size = f->self_size;
            int end = start + self_size;
            set<int> rows_;
            // Self rows
            for (int j = start; j < end; j++)
            {
                rows_.insert(j);
                for (SpMat::InnerIterator it(App, j); it; ++it)
                {
                    int i = it.row();
                    if (i >= start)
                        rows_.insert(i);
                }
            }
            // Add children
            for (auto c : f->cid)
            {
                for (auto i : *(fronts[c]->rows))
                {
                    if (i >= start)
                        rows_.insert(i);
                }
            }
            // Create vector and sort
            auto prows = make_shared<vector<int>>(rows_.begin(), rows_.end());
            sort(prows->begin(), prows->end());
            // Record
            f->nbr_size = prows->size() - self_size;
            f->rows = prows;
            // Create childrens empty data
            int nkids = f->cid.size();
            f->children_s = vector<int>(nkids);
            f->children_n = vector<int>(nkids);
            f->children_rows = vector<pvi>(nkids);
            f->children_fronts = vector<pvd>(nkids);
            f->children_xsol = vector<pvd>(nkids);
        }
        // Allocate fronts
        for (int b = 0; b < nblk; b++)
        {
            int dest = front2rank(b);
            auto f = fronts[b];
            if (dest == upcxx::comm_rank())
            {
                int S = f->rows->size();
                int start = f->start;
                int end = start + f->self_size;
                f->front = make_shared<vector<double>>(S * S, 0.0);
                f->xsol = make_shared<vector<double>>(S, 0.0);
                for (int j = start; j < end; j++)
                {
                    for (SpMat::InnerIterator it(App, j); it; ++it)
                    {
                        int i = it.row();
                        int j = it.col();
                        double v = it.value();
                        if (i >= start && i >= j)
                        {
                            auto row = lower_bound(f->rows->begin(), f->rows->end(), i);
                            assert(row != f->rows->end());
                            int fi = distance(f->rows->begin(), row);
                            int fj = j - start;
                            (*f->front)[fi + fj * S] = v;
                        }
                    }
                }
            }
            else
            {
                fronts.erase(b);
            }
        }
        if (VERB > 0)
            printf("[%d] Symbolic ordering & assembly done, now contains %lu fronts\n", upcxx::comm_rank(), fronts.size());
        if (VERB > 1 && upcxx::comm_rank() == 0)
        {
            for (int i = 0; i < fronts2ranks.size(); i++)
            {
                printf("%d (-> %d) : %d\n", i, treetab[i], fronts2ranks[i]);
            }
        }
    }

    void factorize(int n_threads, int VERB)
    {

        int n_tasks = fronts.size();
        Threadpool tp(n_threads, n_tasks, VERB > 1);
        Taskflow<int> fact_tasks(&tp, VERB > 1);
        fact_tasks_pt = &fact_tasks;
        DepsLogger dlog(1000000);

        fact_tasks.set_compute_on([&](int k) {
                      return (k % n_threads);
                  })
            .set_indegree([&](int k) {
                auto f = fronts.at(k);
                return f->cid.size();
            })
            .set_run([&](int k) {
                auto f = fronts.at(k);
                f->extend_add();
                f->factor();
                if (f->pid != -1)
                {
                    send_front_to_parent(k);
                    dlog.add_event(DepsEvent(fact_tasks.name(k), "Fact_" + to_string(front2rank(f->pid)) + "_" + to_string(f->pid)));
                }
            })
            .set_name([&](int k) {
                return "Fact_" + to_string(upcxx::comm_rank()) + "_" + to_string(k);
            });

        for (auto f : fronts)
        {
            if (f.second->cid.size() == 0)
            {
                fact_tasks.seed(f.first);
            }
        }

        upcxx::barrier();
        timer t0 = wctime();
        tp.start();
        while (!tp.is_done())
        {
            upcxx::progress();
        }
        tp.join();
        upcxx::barrier();
        timer t1 = wctime();
        double tF = elapsed(t0, t1);
        if (VERB > 0)
            printf("[%d] Factorization done, time %3.2e s.\n", upcxx::comm_rank(), tF);
        if (VERB > 0)
        {
            printf("[%d] n_threads n_ranks t_F >>>> %d %d %e\n", upcxx::comm_rank(), n_threads, upcxx::comm_size(), tF);
        }

        std::ofstream logfile;
        logfile.open("dmf_" + to_string(upcxx::comm_rank()) + ".log");
        logfile << tp.get_logger();
        logfile.close();

        std::ofstream depsfile;
        depsfile.open("dmf_" + to_string(upcxx::comm_rank()) + ".dot");
        depsfile << dlog;
        depsfile.close();
    }

    VectorXd solve(VectorXd &b, int n_threads)
    {

        solve_not_done = fronts.size();
        VectorXd xglob = perm.asPermutation() * b;

        int n_tasks = fronts.size() * 3;
        Threadpool tp(n_threads, n_tasks, VERB > 1);
        Taskflow<int> set_tasks(&tp, VERB > 1);
        Taskflow<int> fwd_tasks(&tp, VERB > 1);
        Taskflow<int> bwd_tasks(&tp, VERB > 1);

        fwd_tasks_pt = &fwd_tasks;
        bwd_tasks_pt = &bwd_tasks;

        set_tasks.set_compute_on([&](int k) {
                     return (k % n_threads);
                 })
            .set_indegree([&](int k) {
                return 0;
            })
            .set_run([&](int k) {
                auto f = fronts.at(k);
                f->set_solution(xglob);
                fwd_tasks.fulfill_promise(k);
            })
            .set_name([&](int k) {
                return "SetRhs_" + to_string(upcxx::comm_rank()) + "_" + to_string(k);
            });

        fwd_tasks.set_compute_on([&](int k) {
                     return (k % n_threads);
                 })
            .set_indegree([&](int k) {
                auto f = fronts.at(k);
                return 1 + f->cid.size(); // set_solution + childrens
            })
            .set_run([&](int k) {
                auto f = fronts.at(k);
                f->fwd();
                if (f->pid != -1)
                {
                    send_xsol_to_parent(f->id);
                }
                else
                {
                    bwd_tasks.fulfill_promise(f->id);
                }
            })
            .set_name([&](int k) {
                return "Fwd_" + to_string(upcxx::comm_rank()) + "_" + to_string(k);
            });

        bwd_tasks.set_compute_on([&](int k) {
                     return (k % n_threads);
                 })
            .set_indegree([&](int k) {
                return 1; // parent or, for the roots, itself in the fwd pass
            })
            .set_run([&](int k) {
                auto f = fronts.at(k);
                f->bwd();
                send_xsol_to_children(f->id);
            })
            .set_name([&](int k) {
                return "Bwd_" + to_string(upcxx::comm_rank()) + "_" + to_string(k);
            });

        for (auto f : fronts)
        {
            set_tasks.seed(f.first);
        }

        upcxx::barrier();
        timer t0 = wctime();
        tp.start();
        while (!tp.is_done())
        {
            upcxx::progress();
        }
        tp.join();
        upcxx::barrier();
        timer t1 = wctime();
        if (VERB > 0)
            printf("[%d] Solve done, time %3.2e s.\n", upcxx::comm_rank(), elapsed(t0, t1));

        // Assemble solution from everyone else
        if (upcxx::comm_rank() == 0)
        {
            for (int b = 0; b < nblk; b++)
            {
                fetch_xsol(b, xglob);
            }
        }
        else
        {
            while (solve_not_done > 0)
            {
                upcxx::progress();
            }
        }
        if (VERB > 0)
            printf("[%d] Done solving\n", upcxx::comm_rank());
        return perm.asPermutation().transpose() * xglob;
    }
};

TEST(dmfchol, one)
{
    DistMF lmf(FILENAME, N_LEVELS);
    lmf.factorize(N_THREADS, VERB);
    VectorXd b = VectorXd::Random(lmf.A.rows());
    VectorXd x = lmf.solve(b, N_THREADS);
    if (upcxx::comm_rank() == 0)
    {
        double res = (lmf.A * x - b).norm() / b.norm();
        printf("[%d] |Ax-b|/|b| = %e\n", upcxx::comm_rank(), res);
        ASSERT_LE(res, 1e-10);
    }
}

int main(int argc, char **argv)
{
    upcxx::init();
    ::testing::InitGoogleTest(&argc, argv);
    if (argc >= 2)
    {
        FILENAME = argv[1];
    }
    if (argc >= 3)
    {
        N_LEVELS = atoi(argv[2]);
    }
    if (argc >= 4)
    {
        N_THREADS = atoi(argv[3]);
    }
    if (argc >= 5)
    {
        VERB = atoi(argv[4]);
    }
    string name = get_name_me();
    printf("Rank %d running on %s\n", upcxx::comm_rank(), name.c_str());
    const int return_flag = RUN_ALL_TESTS();
    upcxx::finalize();
    return return_flag;
}
