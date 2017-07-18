#include "function/tanh.h"

namespace heed
{
    namespace function
    {
        template<typename T>
        std::shared_ptr<matrix<T>> tanh<T>::compute(std::shared_ptr<matrix<T>> data)
        {
            // todo: implement me
            return data;
        }
        
        template<typename T>
        std::shared_ptr<matrix<T>> tanh<T>::derivation_slope(std::shared_ptr<matrix<T>> data)
        {
            // todo: implement me
            return data;
        }

        template class tanh<float>;
        template class tanh<double>;
    }
}