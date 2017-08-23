#ifndef Local_h
#define Local_h

#include <cstddef>

using namespace std;

namespace mozart
{
    template<typename T>
    class local
    {
    private:
        size_t _size;
    public:
        local(size_t size)
        {
            this->_size = size;
        }

        size_t memory_size()
        {
            return this->_size * sizeof(T);
        }
    };
}

#endif