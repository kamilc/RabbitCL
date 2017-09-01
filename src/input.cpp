#include "input.h"

namespace mozart
{
    template<typename T>
    activation<T> input<T>::forward(matrix<T> &data)
    {
        return activation<T>(data, false);
    }

    template<typename T>
    activation<T> input<T>::train_forward(matrix<T> &data)
    {
        return activation<T>(data, true);
    }

    template<typename T>
    void input<T>::update_weights(matrix<T>& deltas)
    {
        // no-op
    }

    template<typename T>
    matrix<T>& input<T>::weights()
    {
        throw std::domain_error("Asking for weights on the input layer");
    }

    INSTANTIATE(input);
}