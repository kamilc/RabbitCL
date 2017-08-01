#include "dense.h"

namespace heed
{
    template<typename T, mode MODE>
    dense<T, MODE>::dense(std::size_t size, layer<T, MODE> &input, activation_function<T, MODE> fun) :
        layer<T, MODE>(size, input), _nonlinearity(fun)
    {
        // no-op
    }

    template<typename T, mode MODE>
    matrix<T, MODE> dense<T, MODE>::forward(matrix<T, MODE> &data)
    {
        // first get the computed data from layers below:
        auto in = this->_input->forward(data);

        // next multiply in place by the weights:
        auto out = in.dot(*(this->_weights));

        // normally hold this matrix as it will be needed but for now return it:
        return out;
    }

    template class dense<float, mode::cpu>;
    template class dense<float, mode::gpu>;

    template class dense<double, mode::cpu>;
    template class dense<double, mode::gpu>;
}

