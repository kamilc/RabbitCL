#include "dense.h"

namespace heed
{
    template<typename T, mode MODE>
    dense<T, MODE>::dense(dense_config<T, MODE> &config) :
        layer<T, MODE>(config),
        _fun(config.fun)
    {
        // no-op
    }

    template<typename T, mode MODE>
    matrix<T, MODE> dense<T, MODE>::forward(matrix<T, MODE> &data)
    {
        // // first get the computed data from layers below:
        // auto in = this->_input->forward(data);

        // // next multiply in place by the weights:
        // auto out = in.dot(*(this->_weights));

        // normally hold this matrix as it will be needed but for now return it:

        // todo: implement me
        return data;
    }

    template<typename T, mode MODE>
    dense_config<T, MODE> dense<T, MODE>::with(std::size_t size, typename activation<T, MODE>::function fun)
    {
        return dense_config<T, MODE>(size, fun);
    }

    INSTANTIATE(dense);
}

