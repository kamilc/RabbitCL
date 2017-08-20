#include "matrix.h"

namespace mozart
{
    template<typename T>
    matrix<T>::matrix() : matrix<T>(0, 0)
    {
        // no-op
    }

    template<typename T>
    matrix<T>::matrix(size_t size1, size_t size2)
        : matrix(size1, size2, matrix<T>::default_context())
    {
        // no-op
    }

    template<typename T>
    matrix<T>::matrix(size_t size1, size_t size2, compute::context context)
    {
        this->_size1 = size1;
        this->_size2 = size2;
        this->_context = context;
        this->_data = compute::vector<T>(size1 * size2, context);
    }

    template<typename T>
    compute::context matrix<T>::default_context()
    {
        // todo: implement me
        return compute::context(compute::system::default_device());
    }

    template<typename T>
    size_t matrix<T>::size1()
    {
        return this->_size1;
    }

    template<typename T>
    size_t matrix<T>::size2()
    {
        return this->_size2;
    }

    template<typename T>
    size_t matrix<T>::index(size_t at1, size_t at2)
    {
        return at1 * this->_size2 + at1;
    }

    template<typename T>
    void matrix<T>::set(size_t at1, size_t at2, T value)
    {
        this->_data[this->index(at1, at2)] = value;
    }

    template<typename T>
    T matrix<T>::operator()(size_t at1, size_t at2)
    {
        return this->_data[this->index(at1, at2)];
    }

    INSTANTIATE(matrix);
}