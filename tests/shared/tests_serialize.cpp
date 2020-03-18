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
    test<void(*)(int)>([](int){printf("Test\n");});
    test<double,int,float,s1_t,s2_t>(1.23, 2, 4.32, {5, 6}, {3.14, 2.71, {-1,1}, {-2,2}});
}

TEST(serialize,views) {
    vector<int> d1 = {1, 2, 3, 4, 5};
    vector<double> d2 = {10.10, 11.11, 12.12, 13.13};
    auto v1 = view<int>(d1.data(), d1.size());
    auto v2 = view<double>(d2.data(), d2.size());
    Serializer<view<int>, view<double>> s1;
    Serializer<view<int>, view<double>> s2;
    vector<char> buffer(s1.size(v1, v2));
    s1.write_buffer(buffer.data(), v1, v2);
    auto tup = s2.read_buffer(buffer.data());
    ASSERT_EQ(get<0>(tup).size(), 5);
    ASSERT_EQ(get<1>(tup).size(), 4);
    for(int i = 0; (size_t)i < d1.size(); i++) {
        ASSERT_EQ( *(get<0>(tup).begin() + i), d1[i] );
    }
    for(int i = 0; (size_t)i < d2.size(); i++) {
        ASSERT_EQ( *(get<1>(tup).begin() + i), d2[i] );
    }
}

/**
 * Check that we can serialize message of more than 2^31 (limit of int) bytes
 * We should be able to serialize up to std::numeric_limits<size_t>::max() bytes
 */
TEST(serialize,large) {
    size_t size = static_cast<size_t>(std::numeric_limits<int>::max()) + static_cast<size_t>(478569);
    char* data = (char*)malloc( size * sizeof(char) );
    for(size_t i = 0; i < size; i += static_cast<size_t>(1e5)) {
        data[i] = (i / static_cast<size_t>(17)) % static_cast<size_t>(478965)  + static_cast<size_t>(49); // Anything, really
    }
    ASSERT_TRUE(data != nullptr);
    auto vdata = view<char>(data, size);
    Serializer<view<char>> s;
    char* output = (char*)malloc( s.size(vdata) * sizeof(char) );
    ASSERT_TRUE(output != nullptr);
    s.write_buffer(output, vdata);
    auto tup = s.read_buffer(output);
    view<char>& voutput = get<0>(tup);
    ASSERT_EQ(voutput.size(), vdata.size());
    for(size_t i = 0; i < vdata.size(); i += static_cast<size_t>(1e5)) { // Too slow otherwise
        ASSERT_EQ(vdata.data()[i], voutput.data()[i]);
    }
    free(data);
    free(output);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);    
    const int return_flag = RUN_ALL_TESTS();
    return return_flag;
}
