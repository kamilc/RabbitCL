#include "function/softmax.h"

namespace mozart
{
    namespace function
    {
        // template<typename T, mode MODE>
        // matrix<T, MODE> softmax<T, MODE>::compute(matrix<T, MODE> &data)
        // {
        //     auto diff = data - data.maximum();
        //     auto e_x = matrix<T, MODE>::exp(diff);

        //     auto out = e_x / e_x.sum();

        //     return out;
        // }
        
        // template<typename T, mode MODE>
        // matrix<T, MODE> softmax<T, MODE>::derivation_slope(matrix<T, MODE> &data)
        // {
        //     auto derivation = data - (static_cast<T>(1.0) - data);

        //     return derivation;
        // }

        // INSTANTIATE(softmax);
    }
}