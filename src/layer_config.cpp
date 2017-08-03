#include "layer_config.h"

namespace heed {

    template<typename T, mode MODE>
    void layer_config<T, MODE>::set_input_size(std::size_t size)
    {
        this->_input_size = size;
    }

    template<typename T, mode MODE>
    std::size_t layer_config<T, MODE>::input_size()
    {
        return *this->_input_size;
    }

    template<typename T, mode MODE>
    std::size_t layer_config<T, MODE>::size()
    {
        return this->_size;
    }

    template<typename T, mode MODE>
    bool layer_config<T, MODE>::has_inputs()
    {
        if(this->_input_size) return true;

        return false;
    }

    INSTANTIATE(layer_config);
}