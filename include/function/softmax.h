#ifndef Softmax_h
#define Softmax_h

#include "activation_function.h"
#include "matrix.h"

namespace heed
{
    namespace function
    {
        template<typename T, mode MODE>
        class softmax : public activation_function<T, MODE>
        {
        public:
            std::shared_ptr<matrix<T>> compute(std::shared_ptr<matrix<T>> data);
            std::shared_ptr<matrix<T>> derivation_slope(std::shared_ptr<matrix<T>> data);
        };
    }
}

#endif