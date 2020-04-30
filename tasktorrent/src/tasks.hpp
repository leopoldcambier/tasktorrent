#ifndef __TTOR_SRC_TASKS_HPP__
#define __TTOR_SRC_TASKS_HPP__

#include <functional>
#include <string>

struct Task
{
    std::function<void()> run;
    std::function<void()> fulfill;
    double priority;
    std::string name;
    Task();
    const char *c_name();
};

struct less_pTask
{
    bool operator()(const Task *lhs, const Task *rhs);
};

#endif