#include <viennacl/matrix.hpp>
#include <viennacl/linalg/prod.hpp>

#include "layer.h"

namespace heed
{
    template<typename T, mode MODE>
    layer<T, MODE>::layer(layer_config<T, MODE> &config)
    {
        this->_parent_size = config.input_size();
        this->_size = config.size();
    }

    template<typename T, mode MODE>
    void layer<T, MODE>::initialize_weights()
    {
        if(this->_parent_size)
        {
            auto cols = this->_size;
            auto rows = *this->_parent_size;

            this->_weights = matrix<T, MODE>(rows, cols);
        }
    }

    template<typename T, mode MODE>
    std::size_t layer<T, MODE>::size()
    {
        return this->_size;
    }

    INSTANTIATE(layer);
}

