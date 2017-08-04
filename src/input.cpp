#include "input.h"

namespace mozart
{
    template<typename T>
    input<T>::input(input_config<T> &config) : layer<T>(config)
    {
        // no-op: implemented
    }

    template<typename T>
    matrix<T> input<T>::forward(matrix<T> &data)
    {
        // return calling the copy constructor:
        return data;
    }

    template<typename T>
    input_config<T> input<T>::with(std::size_t size)
    {
        return input_config<T>(size);
    }

    INSTANTIATE(input);
}