#include "layer_config.h"

namespace mozart {

    template<typename T>
    std::size_t layer_config<T>::size()
    {
        return this->_size;
    }

    INSTANTIATE(layer_config);
}