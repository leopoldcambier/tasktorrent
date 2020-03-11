#ifndef __TTOR_VIEWS_HPP__
#define __TTOR_VIEWS_HPP__

namespace ttor {

    template<typename T>
    struct view
    {
        private:
            T* start_; 
            size_t size_;
        public:
            view(T* start, size_t size) : start_(start), size_(size) {};
            view() : start_(nullptr), size_(0) {};
            T* begin() const {
                return start_;
            }
            T* end() const {
                return start_ + size_;
            }
            T* data() const {
                return start_;
            }
            size_t size() const {
                return size_;
            }
            bool operator==(const view<T>& rhs) const {
                return start_ == rhs.start_ && size_ == rhs.size_;
            }
    };
    
}

#endif
