#include "dense.h"

namespace heed
{
    template<typename T, mode MODE>
    dense<T, MODE>::dense(std::size_t size, std::shared_ptr<layer<T, MODE>> input, activation_function<T, MODE> fun) :
        layer<T, MODE>(size, input), _nonlinearity(fun)
    {
        // no-op
    }

    template<typename T, mode MODE>
    std::shared_ptr<dense<T, MODE>> dense<T, MODE>::define(std::size_t size, std::shared_ptr<layer<T, MODE>> input, activation_function<T, MODE> fun)
    {
        return std::make_shared<dense<T, MODE>>(dense<T, MODE>(size, input, fun));
    }

    template<typename T, mode MODE>
    std::shared_ptr<matrix<T, MODE>> dense<T, MODE>::forward(std::shared_ptr<matrix<T, MODE>> data)
    {
        // todo: implement me
        // nonlin(data * weigths)

        return data;
    }

    template class dense<float, mode::cpu>;
    template class dense<float, mode::gpu>;

    template class dense<double, mode::cpu>;
    template class dense<double, mode::gpu>;
}

