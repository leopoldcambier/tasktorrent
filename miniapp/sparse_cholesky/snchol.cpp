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
#include <utility>
#include <queue>

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>

#include <scotch.h>
#include <mpi.h>

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
#include "communications.hpp"
#include "runtime.hpp"

using namespace std;
using namespace Eigen;
using namespace ttor;

typedef Eigen::SparseMatrix<double> SpMat;
typedef array<int, 2> int2;
typedef array<int, 3> int3;

int VERB = 0;
int LOG = 0;
string FILENAME = "neglapl_2_128.mm";
int N_LEVELS = 10;
int N_THREADS = 4;
int BLOCK_SIZE = 10;
string FOLDER = "./";
int REPEAT = 1;

struct range
{
    int lb;
    int ub;
    int k;
};

// Given (k, i, j), returns (k, max(i, j), min(i, j))
int3 lower(int3 kij)
{
    int k = kij[0];
    int i = kij[1];
    int j = kij[2];
    return {k, std::max(i,j), std::min(i,j)};
};

// Find the positions of c_rows into p_rows
//
// @in c_rows is a subset of p_rows
// @in c_rows, p_rows are sorted
// @post out[i] = j <=> c_rows[i] == p_rows[j]
vector<int> get_subids(vector<int> &c_rows, vector<int> &p_rows)
{
    int cn = c_rows.size();
    vector<int> subids(cn);
    int l = 0;
    for (int i = 0; i < cn; i++)
    {
        while (c_rows[i] != p_rows[l])
        {
            l++;
        }
        assert(l < p_rows.size());
        assert(p_rows[l] == c_rows[i]);
        subids[i] = l;
    }
    return subids;
};

// Creates a random (-1, 1) vector
// @pre size >= 0
// @post x, st -1 <= x[i] <= 1 for all 0 <= i < size
VectorXd random(int size, int seed)
{
    mt19937 rng;
    rng.seed(seed);
    uniform_real_distribution<double> dist(-1.0, 1.0);
    VectorXd x(size);
    for (int i = 0; i < size; i++)
    {
        x[i] = dist(rng);
    }
    return x;
}

// Given a sparse matrix A, creates rowval and colptr, the CSC representation of A
// without the self loops on the diagonal
// @pre A is a sparse symmetrix matrix
void get_csc_no_diag(SpMat &A, VectorXi *rowval, VectorXi *colptr) {
    int N = A.rows();
    int nnz = A.nonZeros();
    *rowval = VectorXi(nnz);
    *colptr = VectorXi(N + 1);
    int k = 0;
    (*colptr)[0] = 0;
    for (int j = 0; j < N; j++)
    {
        for (SpMat::InnerIterator it(A, j); it; ++it)
        {
            int i = it.row();
            if (i != j)
            {
                (*rowval)[k] = i;
                k++;
            }
        }
        (*colptr)[j + 1] = k;
    }
    rowval->conservativeResize(k);
}

// Algebraic partitioning
// N is the matrix size
// rowval, colptr is the CSC of symmetric A, without the diagonal
// nlevels, block_size are ND and partitioning parameters
// Outputs are
// - permtab, the permuation 
// - rangtab, the colptr of the clusters (cluster i goes from rangtab[i] ... rangtab[i+1] in permtab ordering)
void algebraic_partitioning(int N, VectorXi &rowval, VectorXi &colptr, int nlevels, int block_size, VectorXi *permtab, VectorXi *rangtab, vector<int> *depth) {
    assert(N == colptr.size() - 1);
    SCOTCH_Graph *graph = SCOTCH_graphAlloc();
    int err = SCOTCH_graphInit(graph);
    assert(err == 0);
    err = SCOTCH_graphBuild(graph, 0, N, colptr.data(), nullptr, nullptr, nullptr, rowval.size(), rowval.data(), nullptr);
    assert(err == 0);
    err = SCOTCH_graphCheck(graph);
    assert(err == 0);
    // Create strat
    SCOTCH_Strat *strat = SCOTCH_stratAlloc();
    err = SCOTCH_stratInit(strat);
    assert(err == 0);
    assert(nlevels > 0);
    string orderingstr = "n{sep=(/levl<" + to_string(nlevels - 1) + "?g:z;),ose=b{cmin=" + to_string(block_size) + "}}";
    cout << "Using ordering " << orderingstr << endl;
    // string orderingstr = "n{sep=(/levl<" + to_string(nlevels-1) + "?g:z;)}";
    err = SCOTCH_stratGraphOrder(strat, orderingstr.c_str());
    assert(err == 0);
    // Order with SCOTCH
    *permtab = VectorXi::Zero(N);
    VectorXi peritab = VectorXi::Zero(N);
    *rangtab = VectorXi::Zero(N + 1);
    VectorXi treetab = VectorXi::Zero(N);
    int nblk = 0;
    err = SCOTCH_graphOrder(graph, strat, permtab->data(), peritab.data(), &nblk, rangtab->data(), treetab.data());
    assert(err == 0);
    assert(nblk >= 0);
    rangtab->conservativeResize(nblk + 1);
    // Compute depth
    *depth = vector<int>(nblk, 0);
    for (int i = 0; i < nblk; i++)
    {
        int p = treetab[i];
        while (p != -1)
        {
            p = treetab[p];
            (*depth)[i]++;
            assert(p == -1 || (p >= 0 && p < nblk));
        }
    }
}

struct Bloc
{
    // row irow of the block
    int i;
    // col irow of the block
    int j;
    // the matrix block
    unique_ptr<MatrixXd> matA;
    // A subset of nodes[i]->start ... nodes[i]->end
    vector<int> rows;
    // A subset of nodes[j]->start ... nodes[j]->end
    vector<int> cols;
    // Accumulation structures
    // number of gemm to accumulate on this block
    int n_accumulate; 
    // data to be accumulated
    mutex to_accumulate_mtx;
    map<int, unique_ptr<MatrixXd>> to_accumulate; // The blocs to be accumulated on this
    // Debugging only
    atomic<bool> accumulating_busy;
    atomic<int> accumulated;
    Bloc(int i_, int j_) : i(i_), j(j_), matA(nullptr), n_accumulate(0), accumulating_busy(false), accumulated(0){};
    MatrixXd* A() { 
        return matA.get(); 
    }
};

struct Node
{
    // irow of this node
    int irow;
    // start, end, size of this node
    int start;
    int end;
    int size;
    // nbrs, after (below the diagonal, in the column) and excluding self
    vector<int> nbrs;
    // used in the solve phase
    VectorXd xsol;
    // children in the etree
    vector<int> children;
    // parent in the etree
    int parent;
    Node(int irow_, int s_, int l_) : irow(irow_), start(s_), size(l_), parent(-1)
    {
        end = start + size;
    };
};

struct DistMat
{
    // All the nodes
    // nodes[i] = ith pivot (diagonal bloc)
    vector<unique_ptr<Node>> nodes;
    // blocs[i,j] = non zero part of the matrix
    map<int2, unique_ptr<Bloc>> blocs;
    // permutation from natural ordering
    VectorXi perm;
    // original A
    SpMat A;
    // App = P A P^T
    SpMat App;
    // Super of supernodes
    int nblk;
    // Map nodes to ranks
    vector<int> node2rank;
    // Depth (in the ND tree) of each node
    vector<int> depth;
    // Map col to rank
    int col2rank(int col)
    {
        assert(node2rank.size() == nblk);
        assert(col >= 0 && col < nblk);
        return node2rank[col];
    }
    // Number of gemm to accumulate on block ij
    int n_to_accumulate(int2 ij)
    {
        return blocs.at(ij)->n_accumulate;
    }
    // Number of gemm accumulated on block ij
    int accumulated(int2 ij)
    {
        return blocs.at(ij)->accumulated.load();
    }
    
    // Some statistics/timings
    atomic<long long> gemm_us;
    atomic<long long> trsm_us;
    atomic<long long> potf_us;
    atomic<long long> scat_us;
    atomic<long long> allo_us;

    // Build all the supernodes based on rangtab
    // Returns i2irow, mapping row -> irow
    VectorXi build_supernodes(VectorXi& rangtab) {
        nodes = vector<unique_ptr<Node>>(nblk);
        int N = App.rows();
        assert(rangtab.size() > 0);
        assert(rangtab[0] == 0);
        assert(nblk == rangtab.size()-1);
        assert(rangtab(nblk) == N);
        VectorXi i2irow(N);
        double mean = 0.0;
        int mini = App.rows();
        int maxi = -1;
        for (int i = 0; i < nblk; i++)
        {
            nodes[i] = make_unique<Node>(i, rangtab[i], rangtab[i + 1] - rangtab[i]);
            for (int j = rangtab[i]; j < rangtab[i + 1]; j++)
            {
                i2irow[j] = i;
            }
            mean += nodes.at(i)->size;
            mini = min(mini, nodes.at(i)->size);
            maxi = max(maxi, nodes.at(i)->size);
        }
        printf("[%d] %d blocks, min size %d, mean size %f, max size %d\n", comm_rank(), nblk, mini, mean / nblk, maxi);
        return i2irow;
    }

    // Build elimination tree, ie, compute 
    // - node->nbrs (irow)
    // - node->parent (irow)
    // - node->children (irow)
    // - block->rows (row)
    // - block->cols (row)
    // row = unknowns in App
    // irow = block #
    // Returns roots, the roots of the elimination tree (usually 1 - not always)
    vector<int> build_tree(VectorXi &i2irow) {
        vector<set<int>> rows_tmp(nblk); // The mapping block -> rows under
        vector<int> roots(0);
        for (int k = 0; k < nblk; k++)
        {
            auto &n = nodes.at(k);
            // Add local rows
            for (int j = n->start; j < n->end; j++)
            {
                for (SpMat::InnerIterator it(App, j); it; ++it)
                {
                    int i = it.row();
                    if (i >= n->end)
                    {
                        rows_tmp.at(k).insert(i);
                    }
                }
            }
            // Get sorted set of neighbros
            vector<int> rows(rows_tmp.at(k).begin(), rows_tmp.at(k).end());
            sort(rows.begin(), rows.end());
            // Convert to neighbors
            set<int> nbrs_tmp;
            for (auto i : rows)
                nbrs_tmp.insert(i2irow(i));
            n->nbrs = vector<int>(nbrs_tmp.begin(), nbrs_tmp.end());
            sort(n->nbrs.begin(), n->nbrs.end());
            // Diagonal bloc
            blocs[{k, k}] = make_unique<Bloc>(k, k);
            auto &b = blocs[{k, k}];
            b->rows = vector<int>(n->size);
            for (int i = 0; i < n->size; i++)
                b->rows[i] = n->start + i;
            b->cols = b->rows;
            // Below-diagonal bloc
            for (auto nirow : n->nbrs)
            {
                auto &nbr = nodes.at(nirow);
                blocs[{nirow, k}] = make_unique<Bloc>(nirow, k);
                auto &b = blocs[{nirow, k}];
                // Find rows
                auto lower = lower_bound(rows.begin(), rows.end(), nbr->start);
                auto upper = upper_bound(rows.begin(), rows.end(), nbr->end - 1);
                b->rows = vector<int>(lower, upper);
                b->cols = blocs.at({k, k})->rows;
            }
            // Add to parent
            if (n->nbrs.size() > 0)
            {
                assert(n->nbrs[0] > k);
                int prow = n->nbrs[0]; // parent in etree = first non zero in column
                n->parent = prow;
                auto &p = nodes.at(prow);
                for (auto i : rows_tmp.at(k))
                {
                    if (i >= p->end)
                    {
                        rows_tmp.at(prow).insert(i);
                    }
                }
                p->children.push_back(k);
                // if(comm_rank() == 0) printf("%d -> %d\n", k, prow);
            }
            else
            {
                n->parent = -1;
                roots.push_back(k);
            }
        }
        printf("%lu roots (should be 1 in general)\n", roots.size());
        return roots;
    }

    // Fill node2rank[k] with a rank for supernode k
    void distribute_tree(vector<int> &roots) {
        queue<int> toexplore;
        vector<range> node2range(nblk);
        for (auto k : roots)
        {
            node2range[k] = {0, comm_size(), 0};
            toexplore.push(k);
        }
        // Distribute tree
        while (!toexplore.empty())
        {
            int k = toexplore.front();
            // if(comm_rank() == 0) printf("exploring %d\n", k);
            auto r = node2range.at(k);
            toexplore.pop();
            auto &n = nodes.at(k);
            if (n->children.size() == 0)
            {
                // Done
            }
            else if (n->children.size() == 1)
            {
                // Same range, just cycle by 1
                auto c = n->children[0];
                assert(r.ub > r.lb);
                int newk = r.lb + (r.k - r.lb + 1) % (r.ub - r.lb);
                node2range[c] = {r.lb, r.ub, newk};
                // if(comm_rank() == 0) printf(" children %d\n", c);
                toexplore.push(c);
            }
            else
            {
                int nc = n->children.size();
                int step = (r.ub - r.lb) / nc;
                // if(comm_rank() == 0) printf("lb ub step nc %d %d %d %d\n", r.lb, r.ub, step, nc);
                if (step == 0)
                { // To many children. Cycle by steps of 1.
                    for (int i = 0; i < nc; i++)
                    {
                        auto c = n->children[i];
                        auto start = r.lb + i % (r.ub - r.lb);
                        auto end = start + 1;
                        assert(start < end);
                        node2range[c] = {start, end, start};
                        toexplore.push(c);
                        // if(comm_rank() == 0) printf(" children %d start %d end %d\n", c, start, end);
                    }
                }
                else
                {
                    for (int i = 0; i < nc; i++)
                    {
                        auto c = n->children[i];
                        auto start = r.lb + i * step;
                        auto end = r.lb + (i + 1) * step;
                        assert(start < end);
                        node2range[c] = {start, end, start};
                        toexplore.push(c);
                        // if(comm_rank() == 0) printf(" children %d start %d end %d\n", c, start, end);
                    }
                }
            }
            // if(comm_rank() == 0) printf(" childrnode2range size %d\n", node2range.size());
        }
        node2rank = vector<int>(nblk, 0);
        for (int i = 0; i < nblk; i++)
        {
            node2rank[i] = node2range.at(i).k;
            // if(comm_rank() == 0)
            // printf("[%d] Node %d - %d\n", comm_rank(), i, node2rank[i]);
        }
    }

    // Allocate blocks[i,k]->A() when column k resides on this rank
    // Fill blocks[i,k]->A() when column k resides on this rank
    void allocate_blocks(VectorXi& i2irow) {
        for (int k = 0; k < nblk; k++)
        {
            if (col2rank(k) == comm_rank())
            {
                // Allocate
                auto &n = nodes.at(k);
                auto &b = blocs.at({k, k});
                b->matA = make_unique<MatrixXd>(n->size, n->size);
                b->A()->setZero();
                for (auto nirow : n->nbrs)
                {
                    auto &b = blocs.at({nirow, k});
                    b->matA = make_unique<MatrixXd>(b->rows.size(), n->size);
                    b->A()->setZero();
                }
                // Fill
                for (int j = n->start; j < n->end; j++)
                {
                    for (SpMat::InnerIterator it(App, j); it; ++it)
                    {
                        int i = it.row();
                        if (i >= n->start)
                        {
                            // Find bloc
                            int irow = i2irow[i];
                            auto &b = blocs.at({irow, k});
                            // Find row
                            auto found = lower_bound(b->rows.begin(), b->rows.end(), i);
                            assert(found != b->rows.end());
                            int ii = distance(b->rows.begin(), found);
                            int jj = j - n->start;
                            (*b->A())(ii, jj) = it.value();
                        }
                    }
                }

            }
        }
    }

    DistMat(std::string filename, int nlevels, int block_size) : nblk(-1), gemm_us(0), trsm_us(0), potf_us(0), scat_us(0), allo_us(0)
    {
        std::cout << "Reading matrix file " << filename << std::endl;
        // Read matrix
        A = mmio::sp_mmread<double, int>(filename);
        // Initialize & prepare
        int N = A.rows();
        VectorXi colptr, rowval;
        get_csc_no_diag(A, &rowval, &colptr);
        // Partition the matrix
        VectorXi rangtab;
        algebraic_partitioning(N, rowval, colptr, nlevels, block_size, &perm, &rangtab, &depth);
        assert(rangtab.size() > 0);
        assert(perm.size() == N);
        nblk = rangtab.size()-1;
        // Permute matrix
        App = perm.asPermutation() * A * perm.asPermutation().transpose();
        // Create all supernodes
        // i2irow maps row/col -> irow (row/col -> block)
        VectorXi i2irow = build_supernodes(rangtab);
        // Compute elimination tree & neighbors
        // roots are the etree roots
        vector<int> roots = build_tree(i2irow);
        // Distribute the tree
        distribute_tree(roots);
        // Allocate blocs
        allocate_blocks(i2irow);
    }

    void print()
    {
        MatrixXd Aff = MatrixXd::Zero(perm.size(), perm.size());
        for (int k = 0; k < nblk; k++)
        {
            auto &n = nodes.at(k);
            int start = n->start;
            int size = n->size;
            Aff.block(start, start, size, size) = blocs.at({k, k})->A()->triangularView<Lower>();
            for (auto i : n->nbrs)
            {
                MatrixXd *Aik = blocs.at({i, k})->A();
                for (int ii = 0; ii < Aik->rows(); ii++)
                {
                    for (int jj = 0; jj < Aik->cols(); jj++)
                    {
                        Aff(blocs.at({i, k})->rows[ii], blocs.at({i, k})->cols[jj]) = (*Aik)(ii, jj);
                    }
                }
            }
        }
        cout << Aff << endl;
    }

    // Factor a diagonal pivot in-place
    void potf(int krow)
    {
        MatrixXd *Ass = blocs.at({krow, krow})->A();
        timer t0 = wctime();
        int err = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', Ass->rows(), Ass->data(), Ass->rows());
        timer t1 = wctime();
        potf_us += (long long)(elapsed(t0, t1) * 1e6);
        assert(err == 0);
    }

    // Trsm a panel bloc in-place
    void trsm(int2 kirow)
    {
        int krow = kirow[0];
        int irow = kirow[1];
        MatrixXd *Ass = blocs.at({krow, krow})->A();
        MatrixXd *Ais = blocs.at({irow, krow})->A();
        timer t0 = wctime();
        cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit,
                    Ais->rows(), Ais->cols(), 1.0, Ass->data(), Ass->rows(), Ais->data(), Ais->rows());
        timer t1 = wctime();
        trsm_us += (long long)(elapsed(t0, t1) * 1e6);
    }

    // Perform a gemm between (i,k) and (j,k) and store the result at (i,j) in to_accumulate
    void gemm(int3 kijrow)
    {
        int krow = kijrow[0];
        int irow = kijrow[1];
        int jrow = kijrow[2];
        MatrixXd *Ais = blocs.at({irow, krow})->A();
        MatrixXd *Ajs = blocs.at({jrow, krow})->A();
        // Do the math
        timer t0, t1, t2;
        t0 = wctime();
        auto Aij_acc = make_unique<MatrixXd>(Ais->rows(), Ajs->rows());
        t1 = wctime();
        if (jrow == irow)
        { // Aii_ = -Ais Ais^T
            cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans,
                        Ais->rows(), Ais->cols(), -1.0, Ais->data(), Ais->rows(), 0.0, Aij_acc->data(), Aij_acc->rows());
        }
        else
        { // Aij_ = -Ais Ajs^T
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                        Ais->rows(), Ajs->rows(), Ais->cols(), -1.0, Ais->data(), Ais->rows(), Ajs->data(), Ajs->rows(), 0.0, Aij_acc->data(), Aij_acc->rows());
        }
        t2 = wctime();
        {
            auto &mtx = blocs.at({irow, jrow})->to_accumulate_mtx;
            auto &acc = blocs.at({irow, jrow})->to_accumulate;
            lock_guard<mutex> lock(mtx);
            acc[krow] = move(Aij_acc);
        }
        allo_us += (long long)(elapsed(t0, t1) * 1e6);
        gemm_us += (long long)(elapsed(t1, t2) * 1e6);
    }

    void accumulate(int3 kijrow)
    {
        int krow = kijrow[0];
        int irow = kijrow[1];
        int jrow = kijrow[2];
        auto &mtx = blocs.at({irow, jrow})->to_accumulate_mtx;
        auto &acc = blocs.at({irow, jrow})->to_accumulate;
        {
            assert(!blocs.at({irow, jrow})->accumulating_busy.load());
            blocs.at({irow, jrow})->accumulating_busy.store(true);
        }
        unique_ptr<MatrixXd> Aij_acc;
        MatrixXd *Aij = blocs.at({irow, jrow})->A();
        timer t0, t1;
        {
            lock_guard<mutex> lock(mtx);
            Aij_acc = move(acc.at(krow));
            acc.erase(acc.find(krow));
        }
        t0 = wctime();
        if (jrow == irow)
        { // Aii_ = -Ais Ais^T
            auto Iids = get_subids(blocs.at({irow, krow})->rows, blocs.at({irow, jrow})->rows);
            for (int j = 0; j < Aij_acc->cols(); j++)
            {
                for (int i = j; i < Aij_acc->rows(); i++)
                {
                    (*Aij)(Iids[i], Iids[j]) += (*Aij_acc)(i, j);
                }
            }
        }
        else
        { // Aij_ = -Ais Ajs^T
            auto Iids = get_subids(blocs.at({irow, krow})->rows, blocs.at({irow, jrow})->rows);
            auto Jids = get_subids(blocs.at({jrow, krow})->rows, blocs.at({irow, jrow})->cols);
            for (int j = 0; j < Aij_acc->cols(); j++)
            {
                for (int i = 0; i < Aij_acc->rows(); i++)
                {
                    (*Aij)(Iids[i], Jids[j]) += (*Aij_acc)(i, j);
                }
            }
        }
        t1 = wctime();
        scat_us += (long long)(elapsed(t0, t1) * 1e6);
        {
            assert(blocs.at({irow, jrow})->accumulating_busy.load());
            blocs.at({irow, jrow})->accumulating_busy.store(false);
        }
    }

    void factorize(int n_threads)
    {
        for (int k = 0; k < nblk; k++) {
            const auto &n = nodes.at(k);
            for (auto i : n->nbrs) {
                for (auto j : n->nbrs) {
                    if (i >= j) {
                        auto &b = blocs.at({i, j});
                        b->n_accumulate++;
                    }
                }
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        timer t0 = wctime();
        printf("Rank %d starting w/ %d threads\n", comm_rank(), n_threads);
        Logger log(1000000);
        Communicator comm(VERB);
        Threadpool tp(n_threads, &comm, VERB, "[" + to_string(comm_rank()) + "]_");
        Taskflow<int> pf(&tp, VERB);
        Taskflow<int2> tf(&tp, VERB);
        Taskflow<int3> gf(&tp, VERB);
        Taskflow<int3> rf(&tp, VERB);
        const int my_rank = comm_rank();

        auto am_send_panel = comm.make_active_msg(
            [&](int &i, int &k, int &isize, int &ksize, view<double> &Aik, view<int> &js) {
                Bloc *b = this->blocs.at({i, k}).get();
                b->matA = make_unique<MatrixXd>(isize, ksize);
                memcpy(b->A()->data(), Aik.data(), Aik.size() * sizeof(double));
                for (auto j : js)
                {
                    gf.fulfill_promise(lower({k, i, j}));
                }
            });

        if (LOG > 0)
        {
            tp.set_logger(&log);
            comm.set_logger(&log);
        }

        pf
            .set_mapping([&](int k) {
                assert(col2rank(k) == my_rank);
                return (k % n_threads);
            })
            .set_indegree([&](int k) {
                assert(col2rank(k) == my_rank);
                int ngemms = n_to_accumulate({k, k});
                return ngemms == 0 ? 1 : ngemms; // # gemms before ?
            })
            .set_task([&](int k) {
                assert(accumulated({k, k}) == n_to_accumulate({k, k}));
                assert(col2rank(k) == my_rank);
                potf(k);
            })
            .set_fulfill([&](int k) {
                assert(accumulated({k, k}) == n_to_accumulate({k, k}));
                assert(col2rank(k) == my_rank);
                auto &n = nodes.at(k);
                for (auto i : n->nbrs)
                {
                    tf.fulfill_promise({k, i});
                }
            })
            .set_name([&](int k) {
                return "[" + to_string(my_rank) + "]_potf_" + to_string(k) + "_lvl" + to_string(depth.at(k));
            })
            .set_priority([](int k) {
                return 3.0;
            });

        tf
            .set_mapping([&](int2 ki) {
                assert(col2rank(ki[0]) == my_rank);
                return (ki[0] % n_threads);
            })
            .set_indegree([&](int2 ki) {
                assert(col2rank(ki[0]) == my_rank);
                int k = ki[0];
                int i = ki[1];
                assert(i > k);
                return n_to_accumulate({i, k}) + 1; // # gemm before + potf
            })
            .set_task([&](int2 ki) {
                assert(col2rank(ki[0]) == my_rank);
                assert(accumulated({ki[1], ki[0]}) == n_to_accumulate({ki[1], ki[0]}));
                trsm(ki);
            })
            .set_fulfill([&](int2 ki) {
                assert(col2rank(ki[0]) == my_rank);
                assert(accumulated({ki[1], ki[0]}) == n_to_accumulate({ki[1], ki[0]}));
                int k = ki[0];
                int i = ki[1];
                Node *n = nodes.at(k).get();
                map<int, vector<int>> deps;
                for (auto j : n->nbrs)
                {
                    int dest = col2rank(lower({k, i, j})[2]);
                    if (dest != my_rank)
                    {
                        deps[dest] = {};
                    }
                }
                for (auto j : n->nbrs)
                {
                    int dest = col2rank(lower({k, i, j})[2]);
                    if (dest != my_rank)
                    {
                        deps[dest].push_back(j);
                    }
                    else
                    {
                        gf.fulfill_promise(lower({k, i, j}));
                    }
                }
                for (auto dep : deps)
                {
                    int dest = dep.first;
                    auto js = dep.second;
                    MatrixXd *Aik = blocs.at({i, k})->A();
                    int isize = Aik->rows();
                    int ksize = Aik->cols();
                    assert(Aik->size() == isize * ksize);
                    auto vAik = view<double>(Aik->data(), Aik->size());
                    auto vJs = view<int>(js.data(), js.size());
                    am_send_panel->named_send(dest, "trsm_" + to_string(k) + "_" + to_string(i),
                                              i, k, isize, ksize, vAik, vJs);
                }
            })
            .set_name([&](int2 ki) {
                return "[" + to_string(my_rank) + "]_trsm_" + to_string(ki[0]) + "_" + to_string(ki[1]) + "_lvl" + to_string(depth[ki[0]]);
            })
            .set_priority([](int2 k) {
                return 2.0;
            });

        gf
            .set_mapping([&](int3 kij) {
                assert(col2rank(kij[2]) == my_rank);
                return (kij[0] % n_threads);
            })
            .set_indegree([&](int3 kij) {
                assert(col2rank(kij[2]) == my_rank);
                int i = kij[1];
                int j = kij[2];
                assert(j <= i);
                return (i == j ? 1 : 2); // Trsms
            })
            .set_task([&](int3 kij) {
                assert(col2rank(kij[2]) == my_rank);
                gemm(kij);
            })
            .set_fulfill([&](int3 kij) {
                assert(col2rank(kij[2]) == my_rank);
                int k = kij[0];
                int i = kij[1];
                int j = kij[2];
                assert(k <= j);
                assert(j <= i);
                // printf("gf %d %d %d -> rf %d %d %d\n", my_rank, k, i, j, k, i, j);
                rf.fulfill_promise(kij);
            })
            .set_name([&](int3 kij) {
                return "[" + to_string(my_rank) + "]_gemm_" + to_string(kij[0]) + "_" + to_string(kij[1]) + "_" + to_string(kij[2]) + "_lvl" + to_string(depth[kij[0]]);
            })
            .set_priority([](int3) {
                return 1.0;
            });

        rf
            .set_mapping([&](int3 kij) {
                assert(col2rank(kij[2]) == my_rank);
                return (kij[1] + kij[2]) % n_threads; // any i & j -> same thread. So k cannot appear in this expression
            })
            .set_indegree([&](int3 kij) {
                assert(col2rank(kij[2]) == my_rank);
                return 1; // The corresponding gemm
            })
            .set_task([&](int3 kij) {
                assert(col2rank(kij[2]) == my_rank);
                blocs.at({kij[1], kij[2]})->accumulated++;
                accumulate(kij);
            })
            .set_fulfill([&](int3 kij) {
                int i = kij[1];
                int j = kij[2];
                if (i == j)
                {
                    // printf("rf %d %d %d -> pf %d\n", k, i, j, i);
                    pf.fulfill_promise(i);
                }
                else
                {
                    // printf("rf %d %d %d -> tf %d %d\n", k, i, j, j, i);
                    tf.fulfill_promise({j, i});
                }
            })
            .set_name([&](int3 kij) {
                return "[" + to_string(my_rank) + "]_acc_" + to_string(kij[0]) + "_" + to_string(kij[1]) + "_" + to_string(kij[2]) + "_lvl" + to_string(depth[kij[0]]);
            })
            .set_priority([](int3) {
                return 4.0;
            })
            .set_binding([](int3) {
                return true;
            });

        for (int k = 0; k < nblk; k++)
        {
            if (col2rank(k) == my_rank)
            {
                if (n_to_accumulate({k, k}) == 0)
                {
                    pf.fulfill_promise(k);
                }
            }
        }

        tp.join();
        MPI_Barrier(MPI_COMM_WORLD);
        printf("Tp & Comms done\n");
        timer t1 = wctime();
        printf("Factorization done, time %3.2e s.\n", elapsed(t0, t1));
        printf("Potf %3.2e s., %3.2e s./thread\n", double(potf_us / 1e6), double(potf_us / 1e6) / n_threads);
        printf("Trsm %3.2e s., %3.2e s./thread\n", double(trsm_us / 1e6), double(trsm_us / 1e6) / n_threads);
        printf("Gemm %3.2e s., %3.2e s./thread\n", double(gemm_us / 1e6), double(gemm_us / 1e6) / n_threads);
        printf("Allo %3.2e s., %3.2e s./thread\n", double(allo_us / 1e6), double(allo_us / 1e6) / n_threads);
        printf("Scat %3.2e s., %3.2e s./thread\n", double(scat_us / 1e6), double(scat_us / 1e6) / n_threads);
        printf(">>>>%d,%d,%d,%3.2e\n", my_rank, comm_size(), n_threads, elapsed(t0, t1));

        auto am_send_pivot = comm.make_active_msg(
            [&](int &k, int &ksize, view<double> &Akk) {
                auto &b = this->blocs.at({k, k});
                b->matA = make_unique<MatrixXd>(ksize, ksize);
                memcpy(b->A()->data(), Akk.data(), Akk.size() * sizeof(double));
            });

        auto am_send_panel2 = comm.make_active_msg(
            [&](int &i, int &k, int &isize, int &ksize, view<double> &Aik) {
                auto &b = this->blocs.at({i, k});
                b->matA = make_unique<MatrixXd>(isize, ksize);
                memcpy(b->A()->data(), Aik.data(), Aik.size() * sizeof(double));
            });

        if (my_rank != 0)
        {
            for (int k = 0; k < nblk; k++)
            {
                if (col2rank(k) == my_rank)
                {
                    // Send column to 0
                    auto &n = nodes.at(k);
                    { // Pivot
                        MatrixXd *Akk = blocs.at({k, k})->A();
                        int ksize = Akk->rows();
                        auto vAkk = view<double>(Akk->data(), Akk->size());
                        am_send_pivot->send(0, k, ksize, vAkk);
                    }
                    for (auto i : n->nbrs)
                    {
                        MatrixXd *Aik = blocs.at({i, k})->A();
                        int ksize = Aik->cols();
                        int isize = Aik->rows();
                        auto vAik = view<double>(Aik->data(), Aik->size());
                        am_send_panel2->send(0, i, k, isize, ksize, vAik);
                    }
                }
            }
        }
        else
        {
            for (int k = 0; k < nblk; k++)
            {
                if (col2rank(k) != 0)
                {
                    comm.recv_process();
                    auto &n = nodes.at(k);
                    for (auto i : n->nbrs)
                        comm.recv_process();
                }
            }
        }

        while(! comm.is_done()) {
            comm.progress();
        }

        if (LOG > 0)
        {
            ofstream logfile;
            string filename = FOLDER + "/snchol_" + to_string(comm_size()) + "_" + to_string(n_threads) + "_" + to_string(App.rows()) + ".log." + to_string(my_rank);
            printf("[%d] Logger saved to %s\n", my_rank, filename.c_str());
            logfile.open(filename);
            logfile << log;
            logfile.close();
        }
    }

    VectorXd solve(VectorXd &b)
    {
        assert(comm_rank() == 0);
        VectorXd xglob = perm.asPermutation() * b;
        // Set solution on each node
        for (int krow = 0; krow < nblk; krow++)
        {
            auto &k = nodes.at(krow);
            k->xsol = xglob.segment(k->start, k->size);
        }
        // Forward
        for (int krow = 0; krow < nblk; krow++)
        {
            auto &k = nodes.at(krow);
            // Pivot xs <- Lss^-1 xs
            MatrixXd *Lss = blocs.at({krow, krow})->A();
            cblas_dtrsv(CblasColMajor, CblasLower, CblasNoTrans, CblasNonUnit, Lss->rows(), Lss->data(), Lss->rows(), k->xsol.data(), 1);
            // Neighbors
            for (int irow : k->nbrs)
            {
                auto &n = nodes.at(irow);
                MatrixXd *Lns = blocs.at({irow, krow})->A();
                VectorXd xn(Lns->rows());
                // xn = -Lns xs
                cblas_dgemv(CblasColMajor, CblasNoTrans, Lns->rows(), Lns->cols(), -1.0, Lns->data(), Lns->rows(), k->xsol.data(), 1, 0.0, xn.data(), 1);
                // Reduce into xn
                auto Iids = get_subids(blocs.at({irow, krow})->rows, blocs.at({irow, irow})->cols);
                for (int i = 0; i < xn.size(); i++)
                {
                    n->xsol(Iids[i]) += xn(i);
                }
            }
        }
        // Backward
        for (int krow = nblk - 1; krow >= 0; krow--)
        {
            auto &k = nodes.at(krow);
            // Neighbors
            for (int irow : k->nbrs)
            {
                auto &n = nodes.at(irow);
                MatrixXd *Lns = blocs.at({irow, krow})->A();
                VectorXd xn(Lns->rows());
                // Fetch from xn
                auto Iids = get_subids(blocs.at({irow, krow})->rows, blocs.at({irow, irow})->cols);
                for (int i = 0; i < xn.size(); i++)
                {
                    xn(i) = n->xsol(Iids[i]);
                }
                // xs -= Lns^T xn
                cblas_dgemv(CblasColMajor, CblasTrans, Lns->rows(), Lns->cols(), -1.0, Lns->data(), Lns->rows(), xn.data(), 1, 1.0, k->xsol.data(), 1);
            }
            // xs = Lss^-T xs
            MatrixXd *Lss = blocs.at({krow, krow})->A();
            cblas_dtrsv(CblasColMajor, CblasLower, CblasTrans, CblasNonUnit, Lss->rows(), Lss->data(), Lss->rows(), k->xsol.data(), 1);
        }
        // Back to x
        for (int krow = 0; krow < nblk; krow++)
        {
            auto &k = nodes.at(krow);
            xglob.segment(k->start, k->size) = k->xsol;
        }
        return perm.asPermutation().transpose() * xglob;
    }
};

void run_cholesky()
{
    printf("[%d] Hello from %s\n", comm_rank(), processor_name().c_str());
    DistMat dm(FILENAME, N_LEVELS, BLOCK_SIZE);
    SpMat A = dm.A;
    dm.factorize(N_THREADS);
    if (comm_rank() == 0)
    {
        VectorXd b = random(A.rows(), 2019);
        VectorXd x = dm.solve(b);
        double res = (A * x - b).norm() / b.norm();
        printf("|Ax-b|/|b| = %e\n", res);
        if(res <= 1e-12) {
            printf("Test ok!");
        } else {
            printf("Error!");
            exit(1);
        }
    }
}

int main(int argc, char **argv)
{
    int req = MPI_THREAD_FUNNELED;
    int prov = -1;
    int err = MPI_Init_thread(NULL, NULL, req, &prov);
    assert(err == 0 && prov == req);
    printf("Usage ./snchol filename nlevels nthreads verb blocksize log folder repeat\n");
    printf("filename = %s, nlevels = %d, nthreads = %d, verb = %d, blocksize = %d, log = %d, folder = %s, repeat = %d\n", 
        FILENAME.c_str(), N_LEVELS, N_THREADS, VERB, BLOCK_SIZE, LOG, FOLDER.c_str(), REPEAT);
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
    if (argc >= 6)
    {
        BLOCK_SIZE = atoi(argv[5]);
    }
    if (argc >= 7)
    {
        LOG = atoi(argv[6]);
    }
    if (argc >= 8)
    {
        FOLDER = argv[7];
    }
    if (argc >= 9)
    {
        REPEAT = atoi(argv[8]);
    }
    for(int r = 0; r < REPEAT; r++) {
        run_cholesky();
    }
    MPI_Finalize();
    return 0;
}
