#include "tasktorrent/tasktorrent.hpp"
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <fstream>
#include <array>
#include <random>
#include <mutex>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <mpi.h>

typedef array<int, 2> int2;
typedef array<int, 3> int3;
typedef array<int, 4> int4;
typedef array<int, 5> int5;
typedef array<int, 6> int6;
typedef array<int, 7> int7;

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
    return to_string(ij[0]) + "_" + to_string(ij[1]);
}

std::string to_string(int3 ijk) {
    return to_string(ijk[0]) + "_" + to_string(ijk[1]) + "_" + to_string(ijk[2]);
}

double val_global(int i, int j) { 
    return static_cast<double>( (i % 49) + (j % 37) * 53 ); 
}