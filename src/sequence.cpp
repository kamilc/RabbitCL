#include "sequence.h"

namespace heed
{
    template<typename T, mode MODE>
    sequence<T, MODE>& sequence<T, MODE>::add(const layer_config<T, MODE> &config)
    {
        // todo: implement me
        return *this;
    }

    template<typename T, mode MODE>
    matrix<T, MODE> sequence<T, MODE>::forward(matrix<T, MODE> &data)
    {
        // todo: implement me
        return data;
    }

    INSTANTIATE(sequence);
}