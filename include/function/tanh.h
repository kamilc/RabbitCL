#ifndef ActivationTanh_h
#define ActivationTanh_h

#include "activation_function.h"
#include "matrix.h"

namespace heed
{
    namespace function
    {
        template<typename T, mode MODE>
        class tanh : public activation_function<T, MODE>
        {
        public:
            matrix<T, MODE> compute(matrix<T, MODE> &data);
            matrix<T, MODE> derivation_slope(matrix<T, MODE> &data);
        };
    }
}

#endif