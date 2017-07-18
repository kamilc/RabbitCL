#ifndef ActivationFunction_h
#define ActivationFunction_h

#include "matrix.h"

namespace heed
{
    template<typename T>
    class activation_function
    {
    public:
        virtual std::shared_ptr<matrix<T>> compute(std::shared_ptr<matrix<T>> data) = 0;
        virtual std::shared_ptr<matrix<T>> derivation_slope(std::shared_ptr<matrix<T>> data) = 0;
    };
}

#endif