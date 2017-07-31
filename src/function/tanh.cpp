#include "function/tanh.h"

namespace heed
{
    namespace function
    {
        template<typename T, mode MODE>
        std::shared_ptr<matrix<T, MODE>> tanh<T, MODE>::compute(std::shared_ptr<matrix<T, MODE>> data)
        {
            // todo: implement me
            return data;
        }
        
        template<typename T, mode MODE>
        std::shared_ptr<matrix<T, MODE>> tanh<T, MODE>::derivation_slope(std::shared_ptr<matrix<T, MODE>> data)
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