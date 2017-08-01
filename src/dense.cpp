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
    void dense<T, MODE>::forward(matrix<T, MODE> &data, matrix<T, MODE> &out)
    {
        // todo: implement me
    }

    template class dense<float, mode::cpu>;
    template class dense<float, mode::gpu>;

    template class dense<double, mode::cpu>;
    template class dense<double, mode::gpu>;
}

