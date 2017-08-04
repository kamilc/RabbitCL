#include "sequence.h"

namespace heed
{
    template<typename T>
    sequence<T>& sequence<T>::add(const layer_config<T> &config)
    {
        // todo: implement me
        return *this;
    }

    template<typename T>
    matrix<T> sequence<T>::forward(matrix<T> &data)
    {
        // todo: implement me
        return data;
    }

    INSTANTIATE(sequence);
}