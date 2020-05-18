#ifndef __TTOR_SRC_VIEWS_HPP__
#define __TTOR_SRC_VIEWS_HPP__

namespace ttor {

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
    T* start_; 
    size_t size_;
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
    view(T* start, size_t size) : start_(start), size_(size) {};

    /**
     * \brief Creates an empty view.
     */
    view() : start_(nullptr), size_(0) {};

    /**
     * \brief The start of the view.
     * 
     * \return A pointer to the start of the view.
     * 
     * \post `begin() + i` for `0 <= i < size()` are all valid addresses for elements of type `T`.
     */
    T* begin() const {
        return start_;
    }

    /**
     * \brief The past-the-end of the view.
     * 
     * \return A pointer to the past-the-end of the view.
     */
    T* end() const {
        return start_ + size_;
    }

    /**
     * \brief The start of the view.
     * 
     * \return A pointer to the start of the view.
     * 
     * \post `data() + i` for `0 <= i < size()` are all valid addresses for elements of type `T`.
     */
    T* data() const {
        return start_;
    }

    /**
     * \brief The size of the view.
     * 
     * \return The number of elements (of type `T`) of the view.
     */
    size_t size() const {
        return size_;
    }

    /**
     * \brief Test views for equality.
     * 
     * \return `true` if both views refer to the same memory buffers with 
     *         the same number of elements; `false` otherwise.
     */
    bool operator==(const view<T>& rhs) const {
        return start_ == rhs.start_ && size_ == rhs.size_;
    }
};
    
} // namespace ttor

#endif
