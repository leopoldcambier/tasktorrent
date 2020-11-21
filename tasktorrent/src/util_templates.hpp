#ifndef __TTOR_SRC_FUNCTIONAL_EXTRA_HPP__
#define __TTOR_SRC_FUNCTIONAL_EXTRA_HPP__

#include <functional>
#include <cstdlib>
#include <type_traits>


namespace ttor {

    // Construct an std::function from a lambda
    // Heavily inspired from https://stackoverflow.com/questions/13358672/how-to-convert-a-lambda-to-an-stdfunction-using-templates
    namespace details {

        template<typename T>
        struct fun_type
        {
            using type = void;
        };

        template<typename Ret, typename Class, typename... Args>
        struct fun_type<Ret(Class::*)(Args...) const>
        {
            using type = std::function<Ret(Args...)>;
        };

        template<typename F>
        typename fun_type<decltype(&F::operator())>::type GetStdFunction(F const &func)
        {
            return func;
        }

    }

    // Construct the type of an active messages based on a std::function
    namespace details {

        template<template<typename T, typename ...Ps> class AM, typename F>
        struct Large_ActiveMsg {
            typedef void type;
        };

        template<template<typename T, typename ...Ps> class AM, typename Ret, typename Class, typename... Args>
        struct Large_ActiveMsg<AM, Ret(Class::*)(Args...) const> {
            typedef AM<std::remove_pointer_t<Ret>,Args...> type;
        };

        template<template<typename T, typename ...Ps> class AM, typename F>
        using Large_AM_t = typename Large_ActiveMsg<AM, F>::type;

        template<template<typename T, typename ...Ps> class AM, typename F>
        struct ActiveMsg {
            typedef void type;
        };

        template<template<typename T, typename ...Ps> class AM, typename Ret, typename Class, typename... Args>
        struct ActiveMsg<AM, Ret(Class::*)(Args...) const> {
            typedef AM<char,Args...> type;
        };

        template<template<typename T, typename ...Ps> class AM, typename F>
        using AM_t = typename ActiveMsg<AM, F>::type;

    }

}

#endif
