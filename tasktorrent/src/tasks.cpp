#include <functional>
#include <string>

#include "tasks.hpp"

Task::Task() : run([](){}), fulfill([](){}), priority(0), name("_") {}

const char *Task::c_name()
{
    return name.c_str();
}

bool less_pTask::operator()(const Task *lhs, const Task *rhs)
{
    return lhs->priority < rhs->priority;
}