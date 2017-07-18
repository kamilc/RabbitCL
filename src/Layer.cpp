#include <viennacl/matrix.hpp>
#include <viennacl/linalg/prod.hpp>

#include "layer.h"

namespace heed
{
    template<typename T>
    layer<T>::layer(std::size_t size)
    {
        this->_size = size;
    }

    template<typename T>
    std::size_t layer<T>::size()
    {
        return this->_size;
    }

    template class layer<float>;
    template class layer<double>;
}

