#ifndef __TTOR_SRC_FUNCTIONAL_EXTRA_HPP__
#define __TTOR_SRC_FUNCTIONAL_EXTRA_HPP__

#include <functional>
#include <cstdlib>

// Heavily inspired from https://stackoverflow.com/questions/13358672/how-to-convert-a-lambda-to-an-stdfunction-using-templates

using namespace std;

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

#endif
