#include "matrix.h"

namespace heed
{
    template<typename T>
    matrix<T>::matrix(mode compMode, std::size_t rows, std::size_t cols, std::vector<T> data)
    {
        // todo: implement me
    }

    template<typename T>
    bool matrix<T>::operator==(const matrix<T> &other)
    {
        // todo: implement me
        return true;
    }

    template class matrix<float>;
}