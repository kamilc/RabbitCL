#include "function/softmax.h"

namespace heed
{
    namespace function
    {
        template<typename T, mode MODE>
        std::shared_ptr<matrix<T, MODE>> softmax<T, MODE>::compute(std::shared_ptr<matrix<T, MODE>> data)
        {
            // todo: implement me
            return data;
        }
        
        template<typename T, mode MODE>
        std::shared_ptr<matrix<T, MODE>> softmax<T, MODE>::derivation_slope(std::shared_ptr<matrix<T, MODE>> data)
        {
            // todo: implement me
            return data;
        }

        template class softmax<float, mode::cpu>;
        template class softmax<float, mode::gpu>;

        template class softmax<double, mode::cpu>;
        template class softmax<double, mode::gpu>;
    }
}