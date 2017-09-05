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
        // std::cout << "Dot with weights: " << this->_weights << std::endl;
        matrix<T> _inter = mozart::function::dot(data, this->_weights);

        return this->_fun(_inter, true);
    }

    template<typename T>
    void dense<T>::update_weights(matrix<T>& deltas)
    {
        this->_weights += deltas;

        // std::cout << "Updating with: " << deltas << std::endl;
        // std::cout << "After updating weights: " << this->_weights << std::endl;
    }

    template<typename T>
    matrix<T>& dense<T>::weights()
    {
        return this->_weights;
    }

    INSTANTIATE(dense);
}

