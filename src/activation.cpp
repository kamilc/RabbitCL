#include "activation.h"
#include "activation_function.h"
#include "function/tanh.h"
#include "function/relu.h"

namespace heed
{
    template<typename T, typename A>
    activation<T, A>::activation(std::shared_ptr<layer<T>> input) : layer<T>(input)
    {
        // no-op
    }

    template<typename T, typename A>
    std::shared_ptr<matrix<T>> activation<T, A>::forward(std::shared_ptr<matrix<T>> data)
    {
        // implement me
        return data;
    }

    template class activation<float, function::tanh<float>>;
    template class activation<float, function::relu<float>>;

    template class activation<double, function::tanh<double>>;
    template class activation<double, function::relu<double>>;
}
