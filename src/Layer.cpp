#include <viennacl/matrix.hpp>
#include <viennacl/linalg/prod.hpp>

#include "layer.h"

namespace heed
{
    template<typename T>
    layer<T>::layer(std::size_t size) : layer<T>(size, std::vector<std::shared_ptr<layer<T>>>())
    {
        // no-op
    }

    template<typename T>
    layer<T>::layer(std::size_t size, std::shared_ptr<layer<T>> input) :
        layer<T>(size, std::vector<std::shared_ptr<layer<T>>>({ input }))
    {
        // no-op
    }

    template<typename T>
    layer<T>::layer(std::shared_ptr<layer<T>> input) :
        layer<T>(input->size(), std::vector<std::shared_ptr<layer<T>>>({ input }))
    {
        // no-op
    }

    template<typename T>
    layer<T>::layer(std::vector<std::shared_ptr<layer<T>>> inputs) :
        layer<T>(inputs[0]->size(), inputs)
    {
        // no-op
        // warning! inputs must not be empty!
    }

    template<typename T>
    layer<T>::layer(std::size_t size, std::vector<std::shared_ptr<layer<T>>> inputs)
    {
        this->_size = size;
        this->_inputs = inputs;
    }

    template<typename T>
    std::size_t layer<T>::size()
    {
        return this->_size;
    }

    template class layer<float>;
    template class layer<double>;
}

