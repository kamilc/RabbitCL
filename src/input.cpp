#include "input.h"
#include "matrix.h"

namespace heed
{
    template<typename T, mode MODE>
    input<T, MODE>::input(input_config<T, MODE> &config) : layer<T, MODE>(config)
    {
        // no-op: implemented
    }

    template<typename T, mode MODE>
    matrix<T, MODE> input<T, MODE>::forward(matrix<T, MODE> &data)
    {
        // return calling the copy constructor:
        return data;
    }

    template<typename T, mode MODE>
    input_config<T, MODE> input<T, MODE>::with(std::size_t size)
    {
        return input_config<T, MODE>(size);
    }

    INSTANTIATE(input);
}