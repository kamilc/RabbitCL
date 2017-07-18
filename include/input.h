#ifndef Input_h
#define Input_h

#include "layer.h"

namespace heed 
{
    template<typename T>
    class input : public layer<T>
    {
    public:
        input(std::size_t size);
        std::shared_ptr<matrix<T>> forward(std::shared_ptr<matrix<T>> data);
    };
}

#endif