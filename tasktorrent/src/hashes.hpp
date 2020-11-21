#ifndef __TTOR_SRC_HASHES_HPP__
#define __TTOR_SRC_HASHES_HPP__

#include <array>
#include <tuple>

namespace ttor {

namespace details {

template <class T>
struct hash_int_N
{
    size_t operator()(const T &t) const
    {
        return std::hash<T>{}(t);
    }
    bool equal(const T &lhs, const T &rhs) const
    {
        return lhs == rhs;
    }
    size_t hash(const T &h) const
    {
        return (*this)(h);
    }
};

template <class T, std::size_t N>
struct hash_int_N<std::array<T, N>>
{

    size_t
    operator()(const std::array<T, N> &t) const
    {

        // Adapted from https://en.cppreference.com/w/cpp/utility/hash/operator()
        // size_t result = 2166136261;
        // std::hash<T> h;
        // for (size_t i = 0; i < N; i++)
        // {
        //     result = result ^ (h(t[i]) << 1);
        // }
        // return result;

        // https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
        // fnv1a hash
        const uint32_t Prime = 0x01000193; //   16777619
        const uint32_t Seed = 0x811C9DC5;  // 2166136261

        uint32_t hash = Seed;

        const unsigned char *ptr = reinterpret_cast<const unsigned char *>(&t[0]);
        int numBytes = 4 * N;

        while (numBytes--)
            hash = (*ptr++ ^ hash) * Prime;

        return hash;
    }

    bool
    equal(const std::array<T, N> &lhs, const std::array<T, N> &rhs) const
    {
        return lhs == rhs;
    }

    size_t hash(const std::array<T, N> &h) const
    {
        return (*this)(h);
    }
};

} // namespace details

} // namespace ttor

#endif