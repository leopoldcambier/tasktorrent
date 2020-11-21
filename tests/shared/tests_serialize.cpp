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

#include "tasktorrent/src/serialization.hpp"

using namespace std;
using namespace ttor;

template<typename... T>
void test(T... t) {
    Serializer<T...> s1;
    Serializer<T...> s2;
    size_t s = s1.size(t...);
    vector<char> buffer(s);
    s1.write_buffer(buffer.data(), s, t...);
    auto tup = s2.read_buffer(buffer.data(), s);
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
    auto v1 = make_view(d1.data(), d1.size());
    auto v2 = make_view(d2.data(), d2.size());
    Serializer<details::view<int>, details::view<double>> s1;
    Serializer<details::view<int>, details::view<double>> s2;
    size_t size = s1.size(v1, v2);
    vector<char> buffer(size);
    s1.write_buffer(buffer.data(), size, v1, v2);
    auto tup = s2.read_buffer(buffer.data(), size);
    ASSERT_EQ((int)get<0>(tup).size(), 5);
    ASSERT_EQ((int)get<1>(tup).size(), 4);
    for(int i = 0; (size_t)i < d1.size(); i++) {
        ASSERT_EQ( *(get<0>(tup).begin() + i), d1[i] );
    }
    for(int i = 0; (size_t)i < d2.size(); i++) {
        ASSERT_EQ( *(get<1>(tup).begin() + i), d2[i] );
    }
}

TEST(serialize,alignement) {
    short a = 42;
    int b = 271;
    double c = 3.14;
    long long d = 4478687;
    vector<int> e = {1, 2, 3};
    vector<double> f = {10.10, 11.11, 12.12};
    Serializer<short,int,double,long long,details::view<int>,details::view<double>> s;
    auto e_v = make_view(e.data(), e.size());
    auto f_v = make_view(f.data(), f.size());
    size_t size = s.size(a,b,c,d,e_v,f_v);
    vector<char> buffer(size);
    s.write_buffer(buffer.data(), buffer.size(), a, b, c, d, e_v, f_v);
    auto tup = s.read_buffer(buffer.data(), buffer.size());
    ASSERT_EQ(get<0>(tup), a);
    ASSERT_EQ(get<1>(tup), b);
    ASSERT_EQ(get<2>(tup), c);
    ASSERT_EQ(get<3>(tup), d);
    for(int i = 0; i < 3; i++) {
        ASSERT_EQ(e[i], *(get<4>(tup).data() + i));
        ASSERT_EQ(f[i], *(get<5>(tup).data() + i));
    }
}

TEST(serialize,alignement2) {
    vector<char> a = {'l', 'o', 'l'};
    vector<int> b = {1, 2, 3, 4};
    vector<double> c = {10.10, 11.11};
    Serializer<details::view<char>,details::view<int>,details::view<double>> s;
    auto a_v = make_view(a.data(), a.size());
    auto b_v = make_view(b.data(), b.size());
    auto c_v = make_view(c.data(), c.size());
    size_t size = s.size(a_v, b_v, c_v);
    vector<char> buffer(size);
    s.write_buffer(buffer.data(), buffer.size(), a_v, b_v, c_v);
    auto tup = s.read_buffer(buffer.data(), buffer.size());
    for(int i = 0; i < 3; i++) {
        ASSERT_EQ(a[i], *(get<0>(tup).data() + i));
    }
    for(int i = 0; i < 4; i++) {
        ASSERT_EQ(b[i], *(get<1>(tup).data() + i));
    }
    for(int i = 0; i < 2; i++) {
        ASSERT_EQ(c[i], *(get<2>(tup).data() + i));
    }
}

TEST(serialize,emptyViews) {
    int z = 0;
    vector<char> a = {};
    vector<int> b = {1};
    vector<double> c = {};
    Serializer<int,details::view<char>,details::view<int>,details::view<double>> s;
    auto a_v = make_view(a.data(), a.size());
    auto b_v = make_view(b.data(), b.size());
    auto c_v = make_view(c.data(), c.size());
    size_t size = s.size(z,a_v, b_v, c_v);
    vector<char> buffer(size);
    s.write_buffer(buffer.data(), buffer.size(), z, a_v, b_v, c_v);
    auto tup = s.read_buffer(buffer.data(), buffer.size());
    ASSERT_EQ((int)get<1>(tup).size(), 0);
    ASSERT_EQ((int)get<2>(tup).size(), 1);
    ASSERT_EQ((int)get<3>(tup).size(), 0);
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
    auto vdata = make_view(data, size);
    Serializer<details::view<char>> s;
    size_t buffer_size = s.size(vdata);
    char* output = (char*)malloc( buffer_size * sizeof(char) );
    ASSERT_TRUE(output != nullptr);
    s.write_buffer(output, buffer_size, vdata);
    auto tup = s.read_buffer(output, buffer_size);
    auto& voutput = get<0>(tup);
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
