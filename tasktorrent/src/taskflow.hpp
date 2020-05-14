#ifndef __TTOR_SRC_TASKFLOW_HPP__
#define __TTOR_SRC_TASKFLOW_HPP__

#include <vector>
#include <unordered_map>
#include <cassert>
#include <functional>
#include <limits>
#include <utility>

#include "util.hpp"
#include "hashes.hpp"

namespace ttor
{

template <class K>
class Taskflow
{

private:
    Threadpool_shared *tp;
    int verb;

    typedef std::unordered_map<K, int, hash_int_N<K>> map_t;
    std::vector<map_t> dep_map;

    std::function<void(K)> f_run;
    std::function<void(K)> f_fulfill;
    std::function<int(K)> f_mapping;
    std::function<bool(K)> f_binding;
    std::function<int(K)> f_indegree;
    std::function<std::string(K)> f_name;
    std::function<double(K)> f_prio;

    // Thread safe
    // Insert task k into dependency map
    void insert(K k)
    {
        int where = -1;
        bool binding;
        auto t = make_task(k, where, binding);
        tp->insert(t, where, binding);
    }

public:

    /**
     * Creates a Taskflow associated with a Threadpool in which task will be inserted.
     * \param tp the associated Threadpool
     * \param verb the verbosity level. 0 is quiet, > 0 is more printing.
     */
    Taskflow(Threadpool_shared *tp_, int verb_ = 0) : tp(tp_), verb(verb_), dep_map(tp_->size())
    {
        f_name = [](K k) { (void)k; return "_"; };
        f_run = [](K k) { (void)k; printf("Taskflow: undefined task function\n"); };
        f_fulfill = [](K k) { (void)k; };
        f_mapping = [](K k) { (void)k; printf("Taskflow: undefined mapping function\n"); return 0; };
        f_binding = [](K k) { (void)k; return false; /* false = migratable [default]; true = bound to thread */ };
        f_indegree = [](K k) { (void)k; printf("Taskflow: undefined indegree function\n"); return 0; };
        f_prio = [](K k) { (void)k; return 0.0; };
    }

    /**
     * Create and return task associated with index `k` and store its mapped thread and binding value.
     * \param k index of the task
     * \param where on what thread should the task be inserted (output)
     * \param binding wether the task should be bound to its thread or not (output)
     * \return A pointer to the new Task. The user is responsible for deleting the task.
     */
    Task *make_task(K k, int &where, bool &binding)
    {
        Task *t = new Task();
        t->run = [this, k]() { f_run(k); };
        t->fulfill = [this, k]() { f_fulfill(k); };
        t->name = f_name(k);
        t->priority = f_prio(k);
        where = f_mapping(k);
        binding = f_binding(k);
        return t;
    }

    /**
     * Set function `f` as the task computational routine
     * \param (void)f(K) the function running the computational task for task `k`
     * \return `this`, the taskflow itself.
     */
    Taskflow &set_task(std::function<void(K)> f)
    {
        f_run = f;
        return *this;
    }

    /**
     * Set function `f` as the task fulfill routine.
     * `set_task` and `set_fulfill` are distinct for logging purposes, but perform exactly the same
     * The fulfill function is always run immediately after the task function
     * \param (void)f(K) the function fulfilling dependencies for task `k`
     * \return `this`, the taskflow itself.
     */
    Taskflow &set_fulfill(std::function<void(K)> f)
    {
        f_fulfill = f;
        return *this;
    }

    /**
     * Set function `f` to be the mapping function, returning the thread of any task `k`
     * \param (int)f(K) the function returning the mapping for task `k`
     * \return `this` the taskflow itself.
     */
    Taskflow &set_mapping(std::function<int(K)> f)
    {
        f_mapping = f;
        return *this;
    }

    /**
     * Set function `f` to be the binding function, returning werther task `k` should be bound to its thread or not
     * \param (bool)f(K) the function returning the binding for task `k`
     * \return `this` the taskflow itself.
     */
    Taskflow &set_binding(std::function<int(K)> f)
    {
        f_binding = f;
        return *this;
    }

    /**
     * Set function `f` to be the indegree function, returning the number of incoming dependencies of task `k`
     * \param (int)f(K) the function returning the in-degree for task `k`
     * \return `this` the taskflow itself.
     */
    Taskflow &set_indegree(std::function<int(K)> f)
    {
        f_indegree = f;
        return *this;
    }

    /**
     * Set function `f` to be the name function, returning a descriptive name for task `k`
     * \param (string)f(K) the function returning the name for task `k`
     * \return `this` the taskflow itself.
     */
    Taskflow &set_name(std::function<std::string(K)> f)
    {
        f_name = f;
        return *this;
    }

    /**
     * Set function `f` to be the priority function, returning the priority (a double) of task `k`
     * When multiple tasks are available for a given thread, tasks with the higher priorities will run first
     * \param (double)f(K) the function returning the priority for task `k`
     * \return `this` the taskflow itself.
     */
    Taskflow &set_priority(std::function<double(K)> f)
    {
        f_prio = f;
        return *this;
    }

    /**
     * Returns the name of task `k`
     * \param k the task index
     * \return task `k` name
     */
    std::string name(K k)
    {
        return f_name(k);
    }

    /**
     * Fulfills a promise for task `k` and decrease dependency count for task with index k.
     * If task cannot be found in the task map, create a new entry.
     * If it exists, reduce by 1 the dependency count.
     * If count == 0, insert the task in the ready queue of its mapped thread.
     * Thread-safe
     * \param k the task index
     */
    void fulfill_promise(K k)
    {
        // Shortcut: if indegree == 1, we can insert the
        // task immediately.
        if (f_indegree(k) == 1)
        {
            insert(k);
            return;
        }

        // We need to create a new entry in the map
        // or decrement the dependency counter.
        const int where = f_mapping(k);
        assert(where >= 0 && where < static_cast<int>(dep_map.size()));

        // Create a task to access and modify the dependency map
        Task *t = new Task();
        t->fulfill = []() {};
        t->name = "dep_map_intern_" + std::to_string(where);
        t->priority = std::numeric_limits<double>::max();

        t->run = [this, where, k]() {
            auto &dmk = this->dep_map[where];

            auto search = dmk.find(k);
            if (search == dmk.end())
            { // k was not found in the map
                // Insert it
                assert(this->f_indegree(k) > 1);
                auto insert_return = dmk.insert(std::make_pair(k, this->f_indegree(k)));
                assert(insert_return.second); // (k,indegree) was successfully inserted

                search = insert_return.first; // iterator for key k
            }

            const int count = --search->second; // decrement dependency counter

            if (count < 0)
            {
                printf("Error: count < 0 for %s\n", name(k).c_str());
                assert(false);
            }

            if (verb > 1)
                printf("%s count: %d\n", name(k).c_str(), count);

            if (count == 0)
            {
                // We erase the entry from the map
                dmk.erase(k);
                insert(k);
            }
        };

        tp->insert(t, where, true);
    }
};

} // namespace ttor

#endif
