#ifndef __TTOR_SRC_VIEWS_HPP__
#define __TTOR_SRC_VIEWS_HPP__

#include <cassert>

namespace ttor {

namespace details {

/**
 * \brief A non-owning view of a memory buffer
 * 
 * \details A view if a pair (buffer, size) representing a view
 *          of a memory buffer. The view does not take ownership
 *          of the buffer; the user is responsible for the buffer
 *          to be valid anytime it is used through the view.
 *          A view's pointer should always be aligned 
 *          (so that `T *start` can be dereferenced).
 */
template<typename T>
struct view
{
private:
    const T* start_; 
    const T* end_;
public:
    /**
     * \brief Creates a view.
     * 
     * \param[in] start A pointer to the start of the buffer of element `T`.
     *                  `start` should be aligned, so that `T *start` is valid (unless `size = 0`).
     * \param[in] size The number of element of type `T` in the buffer.
     * 
     * \pre `size >= 0`
     * \pre if `size > 0`, `start` is a valid pointer of type `T*` and can be dereferenced.
     */
    view(const T* start, size_t size) : start_(start), end_(size > 0 ? start + size : nullptr) {};

    /**
     * \brief Creates a view.
     * 
     * \param[in] start A pointer to the start of the buffer of element `T`.
     *                  `start` should be aligned, so that `T *start` is valid (unless `size = 0`).
     * \param[in] end A pointer to the past-the-end of the buffer of elements `T`
     * 
     */
    view(const T* start, const T* end) : start_(start), end_(end) {
        if(start == nullptr) assert(end == nullptr);
    };

    /**
     * \brief Creates an empty view.
     */
    view() : start_(nullptr), end_(nullptr) {};

    /**
     * \brief The start of the view.
     * 
     * \return A pointer to the start of the view.
     */
    const T* begin() const {
        return start_;
    }

    /**
     * \brief The start of the view.
     * 
     * \return A pointer to the start of the view.
     * 
     * \post view.data() == view.begin()
     */
    const T* data() const {
        return start_;
    }

    /**
     * \brief The past-the-end of the view.
     * 
     * \return A pointer to the past-the-end of the view.
     */
    const T* end() const {
        return end_;
    }

    /**
     * \brief The size of the view
     * 
     * \return If start_ is not null, returns the number of elements in [start_, end_).
     *         Otherwise, returns 0.
     */
    size_t size() const {
        if(start_ == nullptr) {
            return 0;
        } else {
            return std::distance(start_, end_);
        }
    }

    /**
     * \brief Test views for equality.
     * 
     * \return `true` if both views refer to the same memory buffers with 
     *         the same number of elements; `false` otherwise.
     */
    bool operator==(const view<T>& rhs) const {
        return start_ == rhs.start_ && end_ == rhs.end_;
    }
};

} // namespace details

#if defined(TTOR_UPCXX)

#include <upcxx/upcxx.hpp>
template<typename T>
using view = upcxx::view<T, const T*>; // T should not have any cv qualifier (const/volatile) or a be a reference.

#else 

template<typename T>
using view = details::view<T>;

#endif

template<typename T>
view<T> make_view(const T* begin, const T* end) {
#if defined(TTOR_UPCXX)
    if(begin == nullptr) {
        return view<T>();
    } else {
        return upcxx::make_view(begin, end);
    }
#else
    return view<T>(begin, end);
#endif
}

template<typename T>
view<T> make_view(const T* begin, size_t size) {
    if(size == 0) {
        return view<T>();
    } else {
        return make_view(begin, begin + size);
    }
}
    
} // namespace ttor

#endif
