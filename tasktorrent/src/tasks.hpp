#ifndef __TTOR_SRC_TASKS_HPP__
#define __TTOR_SRC_TASKS_HPP__

#include <functional>
#include <string>

/**
 * Task implementation.
 * - `run` and `fulfill` are always called one after the other
 * - `priority` is this task priority. Higher priority tasks are run first when multiple tasks are ready
 * - `name` is a string describind this task
 */
struct Task
{
    std::function<void()> run;
    std::function<void()> fulfill;
    double priority;
    std::string name;
    Task();
    /**
     * \return A C-string giving this tasks name
     */
    const char *c_name();
};

struct less_pTask
{
    bool operator()(const Task *lhs, const Task *rhs);
};

#endif