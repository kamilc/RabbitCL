#include "dense.h"

namespace heed
{
    template<typename T>
    dense<T>::dense(std::size_t size, std::shared_ptr<layer<T>> input) : layer<T>(size, input)
    {
        // no-op
    }

    template<typename T>
    dense<T>::dense(std::size_t size, std::vector<std::shared_ptr<layer<T>>> inputs) : layer<T>(size, inputs)
    {
        // no-op
    }

    template<typename T>
    std::shared_ptr<matrix<T>> dense<T>::forward(std::shared_ptr<matrix<T>> data)
    {
        return data;
    }

    template class dense<float>;
    template class dense<double>;
}

