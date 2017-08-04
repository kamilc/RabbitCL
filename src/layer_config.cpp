#include "layer_config.h"

namespace mozart {

    template<typename T>
    void layer_config<T>::set_input_size(std::size_t size)
    {
        this->_input_size = size;
    }

    template<typename T>
    std::size_t layer_config<T>::input_size()
    {
        return *this->_input_size;
    }

    template<typename T>
    std::size_t layer_config<T>::size()
    {
        return this->_size;
    }

    template<typename T>
    bool layer_config<T>::has_inputs()
    {
        if(this->_input_size) return true;

        return false;
    }

    INSTANTIATE(layer_config);
}