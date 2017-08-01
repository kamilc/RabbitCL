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

        matrix<T, MODE> forward(matrix<T, MODE> &data);
    };
}

#endif