#include "dense.h"

namespace heed
{
    template<typename T>
    dense<T>::dense(dense_config<T> &config) :
        layer<T>(config),
        _fun(config.fun)
    {
        // no-op
    }

    template<typename T>
    matrix<T> dense<T>::forward(matrix<T> &data)
    {
        // // first get the computed data from layers below:
        // auto in = this->_input->forward(data);

        // // next multiply in place by the weights:
        // auto out = in.dot(*(this->_weights));

        // normally hold this matrix as it will be needed but for now return it:

        // todo: implement me
        return data;
    }

    template<typename T>
    dense_config<T> dense<T>::with(std::size_t size, typename activation<T>::function fun)
    {
        return dense_config<T>(size, fun);
    }

    INSTANTIATE(dense);
}

