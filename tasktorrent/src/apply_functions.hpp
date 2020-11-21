#ifndef __TTOR_SRC_APPLY_FUNCTIONS_HPP__
#define __TTOR_SRC_APPLY_FUNCTIONS_HPP__

#include <stddef.h>
#include <utility>
#include <tuple>

namespace ttor {

// Allows calling a function f(A... a) where the data for a is stored in tuple t

namespace detail {

    template <typename T, typename F, size_t... Is>
    constexpr auto apply_impl(F &f, T &t, std::index_sequence<Is...>)
    {
        return f(std::get<Is>(t)...);
    }

    template <typename T, typename F>
    constexpr auto apply(F &f, T &t)
    {
        return apply_impl(f, t, std::make_index_sequence<std::tuple_size<std::decay_t<T>>::value>{});
    }

}

/**
 * \brief Applies a function f(x,y,...) on a tuple (x,y,...)
 */
template <typename T, typename F>
auto apply_fun(F &&fn, T &t)
{
    return detail::apply(fn, t);
}

// Allows calling a function f<V>(V v) on every element of a tuple and return a tuple

namespace detail
{
    template<typename T, typename F, int... Is>
    auto for_each(const T& t, F f, std::integer_sequence<int, Is...>)
    {
        return std::make_tuple(f(std::get<Is>(t))...);
    }
}

/**
 * \brief Applies a templated function on every element of a tuple
 */
template<typename... Ts, typename F>
auto map_tuple(const std::tuple<Ts...> & t, F f)
{
    return detail::for_each(t, f, std::make_integer_sequence<int, sizeof...(Ts)>());
}

} // namespace ttor

#endif