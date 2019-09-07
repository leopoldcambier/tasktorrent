#include <fstream>
#include <array>
#include <random>
#include <exception>
#include <iostream>
#include <sstream>
#include <tuple>
#include <complex>
#include <utility>

#include <gtest/gtest.h>
#include "serialization.hpp"
#include "views.hpp"

using namespace std;
using namespace ttor;

template<typename... T>
void test(T... t) {
    Serializer<T...> s1;
    Serializer<T...> s2;
    vector<char> buffer(s1.size(t...));
    s1.write_buffer(buffer.data(), t...);
    auto tup = s2.read_buffer(buffer.data());
    ASSERT_EQ(tup, make_tuple(t...));
}

struct s1_t {
    long long i;
    short j;
    bool operator==(const s1_t& rhs) const {
        return i == rhs.i && j == rhs.j;
    }
};

struct s2_t {
    double u;
    float  v;
    complex<float> w;
    complex<double> x;
    bool operator==(const s2_t& rhs) const {
        return u == rhs.u && v == rhs.v && w == rhs.w && x == rhs.x;
    }
};

TEST(serialize,all) {
    test<char>('a');
    test<char>('A');
    test<char>('Z');
    test<unsigned char>(0);
    test<unsigned char>(255);
    test<int>(0);
    test<int>(32767);
    test<int>(-32768);
    test<unsigned int>(0);
    test<unsigned int>(65535);
    test<double>(3.141519);
    test<double>(-3.141519);
    test<float>(3.141519);
    test<float>(-3.141519);
    test<int,double,float,char>(1,2.71,-3.14,'A');
    test<void(*)(int)>([](int i){printf("Test\n");});
    test<double,int,float,s1_t,s2_t>(1.23, 2, 4.32, {5, 6}, {3.14, 2.71, {-1,1}, {-2,2}});
}

TEST(serialize,views) {
    vector<int> d1 = {1, 2, 3, 4, 5};
    vector<double> d2 = {10.10, 11.11, 12.12, 13.13};
    auto v1 = view<int>(d1.data(), d1.size());
    auto v2 = view<double>(d2.data(), d2.size());
    Serializer<view<int>, view<double>> s1;
    // , ;
    Serializer<view<int>, view<double>> s2;
    vector<char> buffer(s1.size(v1, v2));
    s1.write_buffer(buffer.data(), v1, v2);
    auto tup = s2.read_buffer(buffer.data());
    ASSERT_EQ(get<0>(tup).size(), 5);
    ASSERT_EQ(get<1>(tup).size(), 4);
    for(int i = 0; i < d1.size(); i++) {
        ASSERT_EQ( *(get<0>(tup).begin() + i), d1[i] );
    }
    for(int i = 0; i < d2.size(); i++) {
        ASSERT_EQ( *(get<1>(tup).begin() + i), d2[i] );
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);    
    const int return_flag = RUN_ALL_TESTS();
    return return_flag;
}
