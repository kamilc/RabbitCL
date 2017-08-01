#ifndef Optimizer_h
#define Optimizer_h

#include "matrix.h"
#include "layer.h"

namespace heed
{
    template<typename T, mode MODE>
    class optimizer
    {
    public:
        virtual void run(layer<T, MODE> &network) = 0;
    };
}

#endif