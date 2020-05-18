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

/**
 * \brief   A parametrized task graph.
 *
 * \details This represents a parametrized task graph.
 *          `K` is the index with which to index tasks.
 * 
 *          Task `k` is defined by
 *          - How many incoming dependencies. See `set_indegree()`.
 *          - What computational function to run, see `set_task()`. 
 *            And what dependencies to fulfill, see `set_fulfill()`.
 *          - On what thread of the associated `Threadpool_shared` should the
 *            task be initially placed. See `set_mapping()`.
 *          - [Optional] the task priority (higher run first).
 *            Default is `0`. See `set_priority()`.
 *          - [Optional] wether the task can be stolen by another
 *            thread. Default is `true`. See `set_binding()`.
 *          
 *          The `Taskflow` is reponsible of maniging dependencies, 
 *          and will automatically run tasks when their dependency count 
 *          reaches 0.
 *          The `Taskflow` is associated to a `Threadpool_shared` in which tasks
 *          will be inserted.
 */
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
     * \brief Creates a Taskflow.
     * 
     * \details A taskflow is associated with a `Threadpool_shared` in which task will be inserted.
     *          `Threadpool_shared` should be valid and not destroyed while the `Taskflow` is used.
     * 
     * \param[in] tp the associated `Threadpool_shared`.
     * \param[in] verb the verbosity level. 0 is quiet, > 0 is more printing.
     * 
     * \pre `verb >= 0`
     * \pre `tp` is a pointer to a valid `Threadpool_shared`.
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
     * \brief Creates a task
     * 
     * \details Creates and returns a task associated with index `k` and store its mapped thread and binding value.
     * 
     * \param[in] k index of the task
     * \param[out] where on what thread should the task be inserted
     * \param[out] binding wether the task should be bound to its thread or not
     * 
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
     * \brief Set function `f` as the task computational routine
     * 
     * \param[in] (void)f(K) the function running the computational task for task `k`
     * 
     * \return `this`, the taskflow itself.
     */
    Taskflow &set_task(std::function<void(K)> f)
    {
        f_run = f;
        return *this;
    }

    /**
     * \brief  Set function `f` as the task fulfill routine.
     * 
     * \details `set_task` and `set_fulfill` are distinct for logging purposes, but perform exactly the same.
     *          The fulfill function is always ran immediately after the task function.
     * 
     * \param[in] (void)f(K) the function fulfilling dependencies for task `k`
     * 
     * \return `this`, the taskflow itself.
     */
    Taskflow &set_fulfill(std::function<void(K)> f)
    {
        f_fulfill = f;
        return *this;
    }

    /**
     * \brief  Set function `f` to be the mapping function, returning the thread of any task `k`.
     * 
     * \param[in] (int)f(K) the function returning the mapping for task `k`.
     * 
     * \return `this` the taskflow itself.
     * 
     * \pre `for all k`, `0 <= f(k) < tp.size()` with `tp` the `Threadpool_shared`.
     */
    Taskflow &set_mapping(std::function<int(K)> f)
    {
        f_mapping = f;
        return *this;
    }

    /**
     * \brief  Set function `f` to be the binding function, returning werther task `k` 
     *         should be bound to its thread (i.e., cannot be stolen) or not.
     * 
     * \param[in] (bool)f(K) the function returning the binding for task `k`.
     * 
     * \return `this` the taskflow itself.
     */
    Taskflow &set_binding(std::function<int(K)> f)
    {
        f_binding = f;
        return *this;
    }

    /**
     * \brief Set function `f` to be the indegree function, returning the number of incoming dependencies of task `k`.
     * 
     * \param[in] (int)f(K) the function returning the in-degree for task `k`
     * 
     * \return `this` the taskflow itself.
     * 
     * \pre `for all k`, `0 <= f(k)`.
     */
    Taskflow &set_indegree(std::function<int(K)> f)
    {
        f_indegree = f;
        return *this;
    }

    /**
     * \brief Set function `f` to be the name function, returning a descriptive name for task `k`.
     * 
     * \param[in] (string)f(K) the function returning the name for task `k`.
     * 
     * \return `this` the taskflow itself.
     */
    Taskflow &set_name(std::function<std::string(K)> f)
    {
        f_name = f;
        return *this;
    }

    /**
     * \brief Set function `f` to be the priority function, returning the priority (a double) of task `k`.
     * 
     * \details When multiple tasks are available for a given thread, tasks with the higher priorities will run first.
     * 
     * \param[in] (double)f(K) the function returning the priority for task `k`.
     * 
     * \return `this` the taskflow itself.
     */
    Taskflow &set_priority(std::function<double(K)> f)
    {
        f_prio = f;
        return *this;
    }

    /**
     * \brief Returns the name of task `k`.
     * 
     * \param[in] k the task index.
     * 
     * \return task `k` name.
     */
    std::string name(K k)
    {
        return f_name(k);
    }

    /**
     * \brief Fulfills a promise on a given task.
     * 
     * \details Fulfills a promise for task `k` and decrease dependency count for task with index k.
     *          If task cannot be found in the task map, create a new entry.
     *          If it exists, reduce by 1 the dependency count.
     *          If count == 0, insert the task in the ready queue of its mapped thread.
     *          Thread-safe.
     * 
     * \param[in] k the task index
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
