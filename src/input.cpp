#include "input.h"

namespace mozart
{
    template<typename T>
    matrix<T> input<T>::forward(matrix<T> &data)
    {
        // return calling the copy constructor:
        return data;
    }

    template<typename T>
    void input<T>::update_weights(matrix<T>& deltas)
    {
        // no-op
    }

    INSTANTIATE(input);
}