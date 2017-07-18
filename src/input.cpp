#include "input.h"
#include "matrix.h"

namespace heed
{
    template<typename T>
    input<T>::input(std::size_t size) : layer<T>(size)
    {
        // no-op
    }

    template<typename T>
    std::shared_ptr<matrix<T>> input<T>::forward(std::shared_ptr<matrix<T>> data)
    {
        return data;
    }

    template class input<float>;
    template class input<double>;
}