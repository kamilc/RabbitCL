#include <viennacl/matrix.hpp>
#include <viennacl/linalg/prod.hpp>

#include "layer.h"

namespace heed
{
    template<typename T, mode MODE>
    layer<T, MODE>::layer(layer<T, MODE> &input) :
        layer<T, MODE>(input.size(), input)
    {
        // no-op
    }

    template<typename T, mode MODE>
    layer<T, MODE>::layer(std::size_t size)
    {
        this->_size = size;
    }

    template<typename T, mode MODE>
    layer<T, MODE>::layer(std::size_t size, layer<T, MODE> &input)
    {
        this->_size = size;
        this->_input = input;
        this->initialize_weights();
    }

    template<typename T, mode MODE>
    void layer<T, MODE>::initialize_weights()
    {
        if(this->_input)
        {
            auto cols = this->_size;
            auto rows = this->_input->size();

            this->_weights = matrix<T, MODE>(rows, cols);
        }
    }

    template<typename T, mode MODE>
    std::size_t layer<T, MODE>::size()
    {
        return this->_size;
    }

    template class layer<float, mode::cpu>;
    template class layer<float, mode::gpu>;

    template class layer<double, mode::cpu>;
    template class layer<double, mode::gpu>;
}

