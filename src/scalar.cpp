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
        this->_data = std::make_shared<compute::array<T, 1>>(context_manager::instance().context());
        (*this->_data)[0] = initial;
    }

    template<typename T>
    void scalar<T>::operator=(T value)
    {
        (*this->_data)[0] = value;
    }

    template<typename T>
    scalar<T>::operator T()
    {
        return (*this->_data)[0];
    }

    template<typename T>
    compute::array<T, 1>& scalar<T>::data()
    {
        return *this->_data;
    }

    INSTANTIATE(scalar)
}
