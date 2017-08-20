#ifndef Optimizer_h
#define Optimizer_h

#include "matrix.h"
#include "utilities.h"
#include "sequence.h"



namespace mozart
{
    template<typename T>
    class optimizer
    {
    public:
        virtual void run(sequence<T> &network, matrix<T> &data, matrix<T> &targets) = 0;
    };
}

#endif