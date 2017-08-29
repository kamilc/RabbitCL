#include "dense.h"

namespace mozart
{
    template<typename T>
    activation<T> dense<T>::forward(matrix<T> &data)
    {
        // todo: implement me
        return activation<T>(data, false);
    }

    template<typename T>
    activation<T> dense<T>::train_forward(matrix<T> &data)
    {
        matrix<T> _inter = mozart::function::dot(data, this->_weights);
        
        return this->_fun(_inter, true);
    }

    template<typename T>
    void dense<T>::update_weights(matrix<T>& deltas)
    {
        // todo: implement me
    }

    INSTANTIATE(dense);
}

