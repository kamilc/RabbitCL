#include "input.h"

namespace mozart
{
    template<typename T>
    matrix<T> input<T>::forward(matrix<T> &data)
    {
        // return calling the copy constructor:
        return data;
    }

    INSTANTIATE(input);
}