#include "layer.h"



namespace mozart
{
    template<typename T>
    std::size_t layer<T>::size()
    {
        return this->_size;
    }

    INSTANTIATE(layer);
}

