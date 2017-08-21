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
    matrix<T>::matrix(size_t size1, size_t size2, compute::context context) :
    matrix(std::make_shared<compute::vector<T>>(size1 * size2, context), 0, size1, 0, size2, context)
    {
        // no-op
    }

    template<typename T>
    matrix<T>::matrix(matrix<T> source, size_t start1, size_t size1, size_t start2, size_t size2) :
      matrix(source._data, start1, size1, start2, size2, source._context)
    {
        // no-op
    }

    template<typename T>
    matrix<T>::matrix(std::shared_ptr<compute::vector<T>> data, size_t start1, size_t size1, size_t start2, size_t size2, compute::context context)
    {
        this->_start1 = start1;
        this->_start2 = start2;
        this->_size1 = size1;
        this->_size2 = size2;
        this->_context = context;
        this->_data = data;
    }

    template<typename T>
    compute::context matrix<T>::default_context()
    {
        // todo: implement me
        return compute::context(compute::system::default_device());
    }

    template<typename T>
    compute::command_queue matrix<T>::default_queue()
    {
        // todo" implement me
        return compute::system::default_queue();
    }

    template<typename T>
    size_t matrix<T>::size1() const
    {
        return this->_transposed ? this->_size2 : this->_size1;
    }

    template<typename T>
    size_t matrix<T>::size2() const
    {
        return this->_transposed ? this->_size1 : this->_size2;
    }

    template<typename T>
    size_t matrix<T>::index(size_t at1, size_t at2)
    {
        // todo: make this work when transposed and when it's a
        // view into a larger matrix:
        return at1 * this->_size2 + at1;
    }

    template<typename T>
    void matrix<T>::set(size_t at1, size_t at2, T value)
    {
        (*this->_data)[this->index(at1, at2)] = value;
    }

    template<typename T>
    void matrix<T>::fill_randn(T mean, T stddev)
    {
        compute::command_queue queue = matrix<T>::default_queue();
        compute::default_random_engine engine(queue);
        compute::normal_distribution<float> distribution(mean, stddev);
        distribution.generate(this->_data->begin(), this->_data->end(), engine, queue);
    }

    template<typename T>
    T matrix<T>::operator()(size_t at1, size_t at2)
    {
        return (*this->_data)[this->index(at1, at2)];
    }

    template<typename T>
    matrix<T> matrix<T>::view(matrix<T> source, size_t start1, size_t end1, size_t start2, size_t end2)
    {
        auto size1 = end1 - start1;
        auto size2 = end2 - start2;

        return matrix<T>(source, start1, size1, start2, size2);
    }

    template<typename T>
    matrix<T> dot(matrix<T>& lhs, matrix<T>& rhs)
    {
        // todo: implement me
        return matrix<T>(lhs.size1(), rhs.size2());
    }

    template<typename T>
    matrix<T> operator*(T lhs, const matrix<T>& rhs)
    {
        // todo: implement me
        return matrix<T>(rhs.size1(), rhs.size2());
    }

    template matrix<float> dot(matrix<float>& lhs, matrix<float>& rhs);
    template matrix<double> dot(matrix<double>& lhs, matrix<double>& rhs);

    template matrix<float> operator*(float lhs, const matrix<float>& rhs);
    template matrix<double> operator*(double lhs, const matrix<double>& rhs);

    INSTANTIATE(matrix);
}