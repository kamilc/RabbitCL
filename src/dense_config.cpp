#include "dense_config.h"
#include "activation.h"

namespace mozart {
    template<typename T>
    dense_config<T>::dense_config(std::size_t size, typename activation<T>::function fun)
    {
        this->_size = size;
        this->fun = fun;
    }

    template<typename T>
    std::shared_ptr<layer<T>> dense_config<T>::construct(std::size_t parent_size) const
    {
        auto _layer = make_shared<dense<T>>();

        _layer->_weights = matrix<T>(parent_size, this->_size);

        // todo: initialize weights randomly with std = 0 and variance of 1

        return _layer;
    }

    INSTANTIATE(dense_config);
}