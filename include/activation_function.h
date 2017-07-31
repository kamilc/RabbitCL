#ifndef ActivationFunction_h
#define ActivationFunction_h

#include "matrix.h"

namespace heed
{
    template<typename T, mode MODE>
    class activation_function
    {
    public:
        virtual std::shared_ptr<matrix<T, MODE>> compute(std::shared_ptr<matrix<T, MODE>> data) = 0;
        virtual std::shared_ptr<matrix<T, MODE>> derivation_slope(std::shared_ptr<matrix<T, MODE>> data) = 0;
    };
}

#endif