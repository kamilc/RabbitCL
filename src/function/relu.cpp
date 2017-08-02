#include "function/relu.h"

namespace heed
{
    namespace function
    {
        template<typename T, mode MODE>
        matrix<T, MODE> relu<T, MODE>::compute(matrix<T, MODE> &data)
        {
            return matrix<T, MODE>::maximum(data, 0);
        }
        
        template<typename T, mode MODE>
        matrix<T, MODE> relu<T, MODE>::derivation_slope(matrix<T, MODE> &data)
        {
            auto signs = matrix<T, MODE>::sign(data);

            return signs.maximum(0);
        }

        template class relu<float, mode::cpu>;
        template class relu<float, mode::gpu>;

        template class relu<double, mode::cpu>;
        template class relu<double, mode::gpu>;
    }
}