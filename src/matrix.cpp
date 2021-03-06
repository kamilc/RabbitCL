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
    matrix(std::make_shared<raw<T>>(size1, size2, context_manager::instance().context()),
          0, size1, size1,
          0, size2, size2)
    {
        // no-op
    }

    template<typename T>
    matrix<T>::matrix(matrix<T>& source, size_t start1, size_t size1, size_t start2, size_t size2) :
      matrix(source._data,
             start1, size1, source._internal_size1,
             start2, size2, source._internal_size2, false)
    {
        // no-op
    }

    template<typename T>
    matrix<T>::matrix(std::shared_ptr<raw<T>> data,
                      size_t start1, size_t size1, size_t internal_size1,
                      size_t start2, size_t size2, size_t internal_size2,
                      bool initialize)
    {
        this->_start1 = start1;
        this->_start2 = start2;
        this->_size1 = size1;
        this->_size2 = size2;
        this->_internal_size1 = internal_size1;
        this->_internal_size2 = internal_size2;
        this->_data = data;

        if(initialize)
        {
          this->fill_zeros();
        }
    }

    template<typename T>
    size_t matrix<T>::size1() const
    {
        return this->_size1;
    }

    template<typename T>
    size_t matrix<T>::size2() const
    {
        return this->_size2;
    }

    template<typename T>
    size_t matrix<T>::offset() const
    {
        return this->_start2 + this->_start1 * this->_internal_size2;
    }

    template<typename T>
    size_t matrix<T>::index(size_t at1, size_t at2) const
    {
        // treating at1 and at2 as starting from 0 and not from start1 and start2
        // returns an internal index within the underlying gpu vector
        auto x = at1;
        auto y = at2;

        return (x + this->_start1) * this->_internal_size2 + y + this->_start2;
    }

    template<typename T>
    void matrix<T>::set(size_t at1, size_t at2, T value)
    {
        (*(*(*this->_data)))[this->index(at1, at2)] = value;
    }

    template<typename T>
    compute::vector<T>& matrix<T>::data() const
    {
        return (*(*(*this->_data)));
    }

    template<typename T>
    void matrix<T>::fill_randn(T mean, T stddev)
    {
        auto queue = context_manager::instance().new_queue();
        vector<T> host_data(this->total_size());

        random_device rnd_device;
        mt19937 mersenne_engine(rnd_device());
        normal_distribution<T> dist(mean, stddev);
        auto gen = std::bind(dist, mersenne_engine);
        generate(host_data.begin(), host_data.end(), gen);

        compute::copy(
            host_data.begin(), host_data.end(), this->data().begin(), queue
        );

        queue.finish();
    }

    template<typename T>
    matrix<T> matrix<T>::reduce_column_sum()
    {
        mozart::opencl::inplace_reduce_column_sum<T>(*this);

        return matrix<T>::view(*this, 0, 0, 0, this->size2() - 1);
    }

    template<typename T>
    void matrix<T>::columnwise_subtract(matrix<T>& other)
    {
      mozart::opencl::inplace_columnwise_subtract<T>(*this, other);
    }

    template<typename T>
    void matrix<T>::fill_zeros()
    {
        auto queue = context_manager::instance().new_queue();
        auto total_size = this->total_size();

        vector<T> host_data(total_size);

        for(size_t i = 0; i < total_size; i++)
        {
          host_data[i] = 0;
        }

        compute::copy(
            host_data.begin(), host_data.end(), this->data().begin(), queue
        );

        queue.finish();
    }

    template<typename T>
    T matrix<T>::operator()(size_t at1, size_t at2)
    {
        return this->data()[this->index(at1, at2)];
    }

    template<typename T>
    matrix<T> matrix<T>::view(matrix<T>& source, size_t start1, size_t end1, size_t start2, size_t end2)
    {
        auto _size1 = end1 + 1 - start1;
        auto _size2 = end2 + 1 - start2;
        auto _start1 = source._start1 + start1;
        auto _start2 = source._start2 + start2;

        assert(start1 <= end1 && start2 <= end2);
        assert(_start1 < source._internal_size1);
        assert(_start2 < source._internal_size2);
        assert(end1 < source._internal_size1);
        assert(end2 < source._internal_size2);

        return matrix<T>(source, _start1, _size1, _start2, _size2);
    }

    template<typename T>
    matrix<T> matrix<T>::clone()
    {
        return matrix<T>(
            std::make_shared<raw<T>>(*(this->_data)),
            this->_start1, this->_size1, this->_internal_size1,
            this->_start2, this->_size2, this->_internal_size2
        );
    }

    template<typename T>
    matrix<T> operator*(T lhs, const matrix<T>& rhs)
    {
        return mozart::opencl::scale<T>(rhs, lhs);
    }

    template<typename T>
    matrix<T> operator+(T lhs, const matrix<T>& rhs)
    {
        return mozart::opencl::scalar_translate<T>(rhs, lhs);
    }

    template<typename T>
    matrix<T> operator+(const matrix<T>& lhs, const matrix<T>& rhs)
    {
        return mozart::opencl::element_add<T>(rhs, lhs);
    }

    template<typename T>
    matrix<T> operator*(const matrix<T>& lhs, const matrix<T>& rhs)
    {
        return mozart::opencl::element_mul<T>(lhs, rhs);
    }

    template<typename T>
    void operator+=(const matrix<T>& lhs, const matrix<T>& rhs)
    {
        mozart::opencl::element_add_assign<T>(lhs, rhs);
    }

    template<typename T>
    void matrix<T>::set_data(std::vector<T>& data)
    {
        auto queue = context_manager::instance().new_queue();

        compute::copy(
            data.begin(),
            data.end(),
            this->data().begin(),
        queue);

        queue.finish();
    }

    template<typename T>
    ostream& operator<<(ostream& os, const matrix<T>& mat)
    {
        matrix_size size = mat.size();
        compute::vector<T>& data = mat.data();

        auto title = (size.internal_size1 != size.size1 || size.internal_size2 != size.size2) ?
            "matrix (view) " : "matrix ";

        os << title << size.size1 << "x" << size.size2 << endl;

        for(auto x = 0; x < size.size1; x++)
        {
            for(auto y = 0; y < size.size2; y++)
            {
                auto _index = mat.index(x, y);
                os <<  data[_index] << " ";
                if(y == size.size2 - 1)
                {
                    os << endl;
                }
            }
        }

        return os;
    }

    template<typename T>
    size_t matrix<T>::total_size() const
    {
        return this->_size1 * this->_size2;
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

        return _size;
    }

    template ostream& operator<<(ostream& os, const matrix<float>& mat);
    template ostream& operator<<(ostream& os, const matrix<double>& mat);

    template matrix<float> operator*(float lhs, const matrix<float>& rhs);
    template matrix<double> operator*(double lhs, const matrix<double>& rhs);

    template matrix<float> operator+(float lhs, const matrix<float>& rhs);
    template matrix<double> operator+(double lhs, const matrix<double>& rhs);

    template matrix<float> operator+(const matrix<float>& lhs, const matrix<float>& rhs);
    template matrix<double> operator+(const matrix<double>& lhs, const matrix<double>& rhs);

    template matrix<float> operator*(const matrix<float>& lhs, const matrix<float>& rhs);
    template matrix<double> operator*(const matrix<double>& lhs, const matrix<double>& rhs);

    template void operator+=(const matrix<float>& lhs, const matrix<float>& rhs);
    template void operator+=(const matrix<double>& lhs, const matrix<double>& rhs);

    INSTANTIATE(matrix);
}
