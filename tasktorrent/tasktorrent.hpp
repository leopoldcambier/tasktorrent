#ifndef __TTOR_TASKTORRENT_HPP__
#define __TTOR_TASKTORRENT_HPP__

/**
 * The public interface to tasktorrent
 * When -DTTOR_SHARED is defined, the distributed part of the library is commented out and is not compiled/included
 */

#include "src/threadpool_shared.hpp"
#include "src/threadpool_dist.hpp"
#include "src/active_messages.hpp"
#include "src/communications.hpp"
#include "src/taskflow.hpp"
#include "src/views.hpp"

#endif