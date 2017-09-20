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
        auto _layer = std::make_shared<dense<T>>();

        // todo: improve the initailization here to help with faster convergence
        _layer->_weights = make_random_matrix<T>(parent_size, this->_size, 0.0, 1.0);
        // std::cout << "Initialized with random matrix: " << _layer->_weights << std::endl;
        _layer->_fun = this->fun;

        return _layer;
    }

    INSTANTIATE(dense_config);
}
