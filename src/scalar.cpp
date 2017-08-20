#include "scalar.h"

namespace mozart
{
    template<typename T>
    scalar<T>::scalar() : scalar(0)
    {
        // no-op
    }

    template<typename T>
    scalar<T>::scalar(T initial)
    {
        this->_data[0] = initial;
    }

    template<typename T>
    void scalar<T>::operator=(T value)
    {
        this->_data[0] = value;
    }

    template<typename T>
    scalar<T>::operator T()
    {
        return this->_data[0];
    }

    INSTANTIATE(scalar)
}