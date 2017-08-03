#ifndef Optimizer_h
#define Optimizer_h

#include "utilities.h"
#include "matrix.h"
#include "sequence.h"

namespace heed
{
    template<typename T, mode MODE>
    class optimizer
    {
    public:
        virtual void run(sequence<T, MODE> &network, matrix<T, MODE> &data, matrix<T, MODE> &targets) = 0;
    };
}

#endif