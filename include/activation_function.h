#ifndef ActivationFunction_h
#define ActivationFunction_h

#include "matrix.h"

namespace heed
{
    template<typename T, mode MODE>
    class activation_function
    {
    public:
        virtual matrix<T, MODE> compute(matrix<T, MODE> &data) = 0;
        virtual matrix<T, MODE> derivation_slope(matrix<T, MODE> &data) = 0;
    };
}

#endif