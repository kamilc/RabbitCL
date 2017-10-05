#include "dense.h"

namespace mozart
{
    template<typename T>
    activation<T> dense<T>::forward(matrix<T> &data)
    {
        matrix<T> _inter = mozart::opencl::dot(data, this->_weights);

        _inter.columnwise_subtract(this->_biases);

        return this->_fun(_inter, false);
    }

    template<typename T>
    activation<T> dense<T>::train_forward(matrix<T> &data)
    {
        matrix<T> _inter = mozart::opencl::dot(data, this->_weights);

        _inter.columnwise_subtract(this->_biases);

        return this->_fun(_inter, true);
    }

    template<typename T>
    void dense<T>::update_weights(matrix<T>& deltas)
    {
        this->_weights += deltas;
    }

    template<typename T>
    void dense<T>::update_bias(matrix<T>& deltas)
    {
        this->_biases += deltas.reduce_column_sum();
    }

    template<typename T>
    matrix<T>& dense<T>::weights()
    {
        return this->_weights;
    }

    INSTANTIATE(dense);
}

