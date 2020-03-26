#ifndef __TTOR_SERIALIZATION_HPP__
#define __TTOR_SERIALIZATION_HPP__

#include <tuple>
#include <functional>
#include <vector>
#include <utility>
#include <cstring>
#include <cassert>
#include <memory>

#include "views.hpp"

template <std::size_t... Ns, typename... Ts>
constexpr auto tail_impl(std::index_sequence<Ns...>, std::tuple<Ts...> t)
{
    return std::make_tuple(std::get<Ns + 1u>(t)...);
}

template <typename... Ts>
constexpr auto tail(std::tuple<Ts...> t)
{
    return tail_impl(std::make_index_sequence<sizeof...(Ts) - 1u>(), t);
}

namespace ttor
{

// Serialize data
template <typename... Values>
class Serializer
{
private:
    // Allows to call a function for each element of a tuple
    template <typename T, typename F, size_t... Is>
    constexpr void for_each_impl(T &t, F &&f, std::index_sequence<Is...>)
    {
        auto l = {(f(std::get<Is>(t)), 0)...};
        (void)l;
    }
    template <typename... Ts, typename F>
    constexpr void for_each_in_tuple(std::tuple<Ts...> &t, F &&f)
    {
        for_each_impl(t, std::forward<F>(f), std::index_sequence_for<Ts...>{});
    }

    // Computing the buffer size
    class for_size_buffer
    {
    public:
        size_t total;
        for_size_buffer() : total(0) {}
        template <typename T>
        void operator()(T&)
        {
            total += sizeof(T);
        }
        template <typename T>
        void operator()(view<T> &t)
        {
            total += sizeof(size_t);
            total += (total % alignof(T));
            total += sizeof(T) * t.size();
        }
    };

    // Write data from tuple to buffer
    class for_write_buffer
    {
    private:
        char *p;
        size_t size;

    public:
        for_write_buffer(char *b, size_t s) : p(b), size(s) {}
        template <typename T>
        void operator()(T &t)
        {
            memcpy(p, &t, sizeof(T));
            p += sizeof(T);
            size -= sizeof(T);
            assert(size >= 0);
        }
        template <typename T>
        void operator()(view<T> &t)
        {
            // Serialize the size
            size_t length = t.size();
            memcpy(p, &length, sizeof(size_t));
            p += sizeof(size_t);
            size -= sizeof(size_t);
            assert(size >= 0);
            // Align
            void* pv = (void*)p;
            void* error = std::align(alignof(T), sizeof(T), pv, size);
            p = (char*)pv;
            assert(error != nullptr);
            assert(size >= 0);
            // Serialize the data at the now aligned address
            size_t size_view = length * sizeof(T);
            memcpy(p, t.data(), size_view);
            p += size_view;
            size -= size_view;
            assert(size >= 0);
        }
        // TODO: Specialize for other types
    };

    // Read data from buffer into a tuple
    class for_read_buffer
    {
    private:
        char *p;
        size_t size;

    public:
        for_read_buffer(char *b, size_t s) : p(b), size(s) {}
        template <typename T>
        void operator()(T &t)
        {
            memcpy(&t, p, sizeof(T));
            p += sizeof(T);
            size -= sizeof(T);
            assert(size >= 0);
        }
        template <typename T>
        void operator()(view<T> &t)
        {
            // Deserialize size
            size_t length;
            memcpy(&length, p, sizeof(size_t));
            p += sizeof(size_t);
            size -= sizeof(T);
            assert(size >= 0);
            // Align
            void* pv = (void*)p;
            void* error = std::align(alignof(T), sizeof(T), pv, size);
            p = (char*)pv;
            assert(error != nullptr);
            assert(size >= 0);
            // Deserialize data - avoid unnecessary copy and fetch the (aligned) pointer
            size_t size_view = length * sizeof(T);
            T *start = reinterpret_cast<T *>(p);
            t = view<T>(start, length);
            p += size_view;
            size -= size_view;
            assert(size >= 0);
        }
    };

public:
    size_t size(Values &... vals)
    {
        for_size_buffer sizer;
        auto tup = std::make_tuple(vals...);
        for_each_in_tuple(tup, sizer);
        return sizer.total;
    }

    void write_buffer(char *buffer, size_t size, Values &... vals)
    {
        auto tup = std::make_tuple(vals...);
        for_each_in_tuple(tup, for_write_buffer(buffer, size));
    }

    std::tuple<Values...> read_buffer(char *buffer, size_t size)
    {
        std::tuple<Values...> tup;
        for_each_in_tuple(tup, for_read_buffer(buffer, size));
        return tup;
    }
};

template <>
class Serializer<>
{
public:
    size_t size() { return 0; };
    void write_buffer(char *){};
    std::tuple<> read_buffer(char *) { return {}; };
};

void print_bytes(const void *ptr, size_t size);
void print_bytes(std::vector<char> buffer);
} // namespace ttor

#endif
