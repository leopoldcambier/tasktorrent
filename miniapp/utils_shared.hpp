#include "tasktorrent/tasktorrent.hpp"
#include <Eigen/Core>
#include <Eigen/Cholesky>
#ifdef USE_MKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif
#include <fstream>
#include <array>
#include <random>
#include <mutex>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <cxxopts.hpp>

typedef std::array<int, 2> int2;
typedef std::array<int, 3> int3;
typedef std::array<int, 4> int4;
typedef std::array<int, 5> int5;
typedef std::array<int, 6> int6;
typedef std::array<int, 7> int7;

struct scoped_timer {
  private:
    std::atomic<long long int>* time_us_;
    ttor::timer time_init_;
  public:
    scoped_timer(std::atomic<long long int>* time_us) {
        time_us_ = time_us;
        time_init_= ttor::wctime();
    }
    ~scoped_timer() {
        ttor::timer time_end_ = ttor::wctime();
        *time_us_ += static_cast<long long int>(1e6 * ttor::elapsed(time_init_, time_end_));
    }
};

ttor::view<double> make_view(Eigen::MatrixXd* A) {
    return ttor::view<double>(A->data(), A->size());
}

Eigen::MatrixXd make_from_view(ttor::view<double> A, int nrows) {
    Eigen::MatrixXd Add = Eigen::MatrixXd::Zero(nrows, nrows);
    assert(nrows * nrows == A.size());
    memcpy(Add.data(), A.data(), sizeof(double) * A.size());
    return Add;
}

void copy_from_view(Eigen::MatrixXd* dest, const ttor::view<double> A) {
    assert(dest->size() == A.size());
    memcpy(dest->data(), A.data(), sizeof(double) * A.size());
}

void accumulate(Eigen::MatrixXd* dest, const Eigen::MatrixXd* src) {
    assert(dest->size() == src->size());
    #pragma omp parallel for
    for(int k = 0; k < dest->size(); k++) {
        (*dest)(k) += (*src)(k);
    }
}

std::string to_string(int2 ij) {
    return std::to_string(ij[0]) + "_" + std::to_string(ij[1]);
}

std::string to_string(int3 ijk) {
    return std::to_string(ijk[0]) + "_" + std::to_string(ijk[1]) + "_" + std::to_string(ijk[2]);
}

double val_global(int i, int j) { 
    return static_cast<double>( (i % 49) + (j % 37) * 53 ); 
}

void warmup_mkl(int n_threads) {
    auto comm = ttor::make_communicator_world();
    ttor::Threadpool tp(n_threads, comm.get());
    ttor::Taskflow<int> warmup(&tp);
    
    warmup.set_task([&](int i){
        Eigen::MatrixXd A = Eigen::MatrixXd::Identity(256,256);
        Eigen::MatrixXd B = Eigen::MatrixXd::Identity(256,256);
        Eigen::MatrixXd C = Eigen::MatrixXd::Identity(256,256);
    	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 256, 256, 256, 1.0, A.data(), 256, B.data(), 256, 1.0, C.data(), 256);
    }).set_indegree([&](int i) {
        return 1;
    }).set_mapping([&](int i) {
        return i % n_threads;
    });

    for(int i = 0; i < 64; i++) {
        warmup.fulfill_promise(i);
    }
    tp.join();
    ttor::comms_world_barrier();
}
