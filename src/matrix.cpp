#include "matrix.h"

namespace mozart
{
    template<typename T>
    matrix<T>::matrix() : matrix<T>(0, 0)
    {
        // no-op
    }

    template<typename T>
    matrix<T>::matrix(size_t size1, size_t size2) :
    matrix(std::make_shared<compute::vector<T>>(size1 * size2, context_manager::instance().context()),
          0, size1, size1,
          0, size2, size2)
    {
        // no-op
    }

    template<typename T>
    matrix<T>::matrix(matrix<T> source, size_t start1, size_t size1, size_t start2, size_t size2) :
      matrix(source._data,
             start1, size1, source._internal_size1,
             start2, size2, source._internal_size2)
    {
        // no-op
    }

    template<typename T>
    matrix<T>::matrix(std::shared_ptr<compute::vector<T>> data,
                      size_t start1, size_t size1, size_t internal_size1,
                      size_t start2, size_t size2, size_t internal_size2)
    {
        this->_start1 = start1;
        this->_start2 = start2;
        this->_size1 = size1;
        this->_size2 = size2;
        this->_internal_size1 = internal_size1;
        this->_internal_size2 = internal_size2;
        this->_data = data;
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
    size_t matrix<T>::index(size_t at1, size_t at2) const
    {
        // todo: make this work when transposed and when it's a
        // view into a larger matrix:
        return at1 * this->_size2 + at2;
    }

    template<typename T>
    void matrix<T>::set(size_t at1, size_t at2, T value)
    {
        (*this->_data)[this->index(at1, at2)] = value;
    }

    template<typename T>
    compute::vector<T>& matrix<T>::data() const
    {
        return *this->_data;
    }

    template<typename T>
    void matrix<T>::fill_randn(T mean, T stddev)
    {
        auto queue = context_manager::instance().new_queue();

        compute::default_random_engine engine(queue);
        compute::normal_distribution<float> distribution(mean, stddev);
        distribution.generate(this->_data->begin(), this->_data->end(), engine, queue);

        queue.finish();
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
        matrix<T> out(lhs.size1(), rhs.size2());

        // run<T>(kernel<dot<T>>::instance(), lhs, rhs, out);

        return out;
    }

    template<typename T>
    matrix<T> operator*(T lhs, const matrix<T>& rhs)
    {
        // todo: implement me
        return matrix<T>(rhs.size1(), rhs.size2());
    }

    template<typename T>
    void matrix<T>::set_data(std::vector<T>& data)
    {
        auto queue = context_manager::instance().new_queue();

        compute::copy(
            data.begin(),
            data.end(),
            this->_data->begin(),
        queue);

        queue.finish();
    }

    template<typename T>
    ostream& operator<<(ostream& os, const matrix<T>& mat)
    {
        matrix_size size = mat.size();
        compute::vector<T>& data = mat.data();

        os << "matrix " << size.size1 << "x" << size.size2 << endl;

        for(auto x = size.start1; x < size.start1 + size.size1; x++)
        {
            for(auto y = size.start2; y < size.start2 + size.size2; y++)
            {
                os << data[mat.index(x, y)] << " ";
                if(y == size.size2 - 1)
                {
                    os << endl;
                }
            }
        }

        return os;
    }

    template<typename T>
    matrix_size matrix<T>::size() const
    {
        matrix_size _size;

        _size.size1 = this->_size1;
        _size.size2 = this->_size2;
        _size.internal_size1 = this->_internal_size1;
        _size.internal_size2 = this->_internal_size2;
        _size.start1 = this->_start1;
        _size.start2 = this->_start2;
        _size.transposed = this->_transposed;

        return _size;
    }

    template ostream& operator<<(ostream& os, const matrix<float>& mat);
    template ostream& operator<<(ostream& os, const matrix<double>& mat);

    template matrix<float> dot(matrix<float>& lhs, matrix<float>& rhs);
    template matrix<double> dot(matrix<double>& lhs, matrix<double>& rhs);

    template matrix<float> operator*(float lhs, const matrix<float>& rhs);
    template matrix<double> operator*(double lhs, const matrix<double>& rhs);

    INSTANTIATE(matrix);
}