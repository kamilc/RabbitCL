#include "layer.h"

using namespace viennacl;

namespace mozart
{
    // template<typename T>
    // layer<T>::layer(layer_config<T> &config)
    // {
    //     this->_parent_size = config.input_size();
    //     this->_size = config.size();
    // }

    template<typename T>
    void layer<T>::initialize_weights()
    {
        if(this->_parent_size)
        {
            auto cols = this->_size;
            auto rows = *this->_parent_size;

            this->_weights = matrix<T>(rows, cols);
        }
    }

    template<typename T>
    std::size_t layer<T>::size()
    {
        return this->_size;
    }

    INSTANTIATE(layer);
}

