#include "serialization.hpp"

namespace ttor {
   
    void print_bytes(const void* ptr, size_t size) {
        const char* ptr_bytes = reinterpret_cast<const char*>(ptr);
        for(size_t i = 0; i < size; i++){
            printf("%02X ", (unsigned char)ptr_bytes[i]);
        }
        printf("\n");
    }

    void print_bytes(std::vector<char> buffer) {
        print_bytes(buffer.data(), buffer.size());
    }
    
}
