#ifndef __TTOR_SRC_COMMUNICATIONS_HPP__
#define __TTOR_SRC_COMMUNICATIONS_HPP__

#include <utility>
#include <mutex>
#include <memory>
#include <functional>
#include <list>
#include <map>
#include <type_traits>

#include "util.hpp"

namespace ttor
{

/**
 * \brief Handles all inter-ranks communications.
 * 
 * \details Object responsible for communications accross ranks.
 *          - All MPI calls will be funelled through this object
 *          - UPC++ calls are processed by this object on the receiver
 */
class Communicator_Base
{

protected:

    const int verb;
    bool log;
    Logger *logger;

public:

    /**
     * \brief Creates a communicator
     * 
     * \param[in] verb the verbosity level. 0 is quiet, > 0 is more and more chatty.
     */
    Communicator_Base(int verb);

    /**
     * \brief Sets the logger.
     * 
     * \details Not thread safe
     * 
     * \param[in] logger a pointer to the logger. The logger is not owned by `this`.
     * 
     * \pre `logger` should be a pointer to a valid `Logger`, that should not be destroyed while `this` is in use.
     */
    void set_logger(Logger *logger);

    /**
     * \brief Makes progress on the communications.
     * 
     * \details Should be called from the main thread (the one that called `MPI_Init_Thread` or `upcxx::init`)
     *          Not thread safe.
     */
    virtual void progress() = 0;

    /**
     * \brief Checks for local completion.
     * 
     * \return `true` if all queues are empty, `false` otherwise.
     */
    virtual bool is_done() const = 0;

    /**
     * \brief Number of locally processed active messages.
     * 
     * \details An active message is processed on the receiver when the associated LPC has finished running.
     * 
     * \return The number of processed active message. 
     */
    virtual llint get_n_msg_processed() const = 0;

    /**
     * \brief Number of locally queued active messages.
     * 
     * \details An active message is queued on the sender after a call to `am->send(...)`.
     * 
     * \return The number of queued active message.
     */
    virtual llint get_n_msg_queued() const = 0;

    /**
     * \brief The rank within the communicator
     * 
     * \return The MPI/UPC++ rank of the current processor within its communicator/team
     */
    virtual int comm_rank() const = 0;

    /**
     * \brief The size of the communicator
     * 
     * \return The number of MPI/UPC++ ranks within the communicator/team
     */
    virtual int comm_size() const = 0;

    /**
     * \brief Destroys the communicator
     */
    virtual ~Communicator_Base();
};

} // namespace ttor

#endif