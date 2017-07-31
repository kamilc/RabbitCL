#ifndef Input_h
#define Input_h

#include "layer.h"

namespace heed 
{
    template<typename T, mode MODE>
    class input : public layer<T, MODE>
    {
    public:
        input(std::size_t size);
        std::shared_ptr<matrix<T, MODE>> forward(std::shared_ptr<matrix<T, MODE>> data);

        static std::shared_ptr<input<T, MODE>> define(std::size_t size);
    };
}

#endif