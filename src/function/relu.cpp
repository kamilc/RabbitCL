#include "function/relu.h"

namespace heed
{
    namespace function
    {
        template<typename T, mode MODE>
        std::shared_ptr<matrix<T>> relu<T, MODE>::compute(std::shared_ptr<matrix<T>> data)
        {
            // todo: implement me
            return data;
        }
        
        template<typename T, mode MODE>
        std::shared_ptr<matrix<T>> relu<T, MODE>::derivation_slope(std::shared_ptr<matrix<T>> data)
        {
            // todo: implement me
            return data;
        }

        template class relu<float, mode::cpu>;
        template class relu<float, mode::gpu>;

        template class relu<double, mode::cpu>;
        template class relu<double, mode::gpu>;
    }
}