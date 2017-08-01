#include "function/tanh.h"

namespace heed
{
    namespace function
    {
        template<typename T, mode MODE>
        matrix<T, MODE> tanh<T, MODE>::compute(matrix<T, MODE> &data)
        {
            // todo: implement me
            return data;
        }
        
        template<typename T, mode MODE>
        matrix<T, MODE> tanh<T, MODE>::derivation_slope(matrix<T, MODE> &data)
        {
            // todo: implement me
            return data;
        }

        template class tanh<float, mode::cpu>;
        template class tanh<float, mode::gpu>;

        template class tanh<double, mode::cpu>;
        template class tanh<double, mode::gpu>;
    }
}