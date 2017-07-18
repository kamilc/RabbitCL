#ifndef ActivationReLU_h
#define ActivationReLU_h

#include "activation_function.h"
#include "matrix.h"

namespace heed
{
    namespace function
    {
        template<typename T>
        class relu : public activation_function<T>
        {
        public:
            std::shared_ptr<matrix<T>> compute(std::shared_ptr<matrix<T>> data);
            std::shared_ptr<matrix<T>> derivation_slope(std::shared_ptr<matrix<T>> data);
        };
    }
}

#endif