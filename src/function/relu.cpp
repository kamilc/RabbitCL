#include "function/relu.h"

namespace heed
{
    namespace function
    {
        template<typename T>
        std::shared_ptr<matrix<T>> relu<T>::compute(std::shared_ptr<matrix<T>> data)
        {
            // todo: implement me
            return data;
        }
        
        template<typename T>
        std::shared_ptr<matrix<T>> relu<T>::derivation_slope(std::shared_ptr<matrix<T>> data)
        {
            // todo: implement me
            return data;
        }

        template class relu<float>;
        template class relu<double>;
    }
}