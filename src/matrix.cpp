#include "matrix.h"

using namespace arma;

namespace heed
{
    template<typename T>
    class matrix<T, mode::cpu> : public matrix_base<T, mode::cpu> {
    private:
        Mat<T> _data;
    public:
        matrix(std::size_t rows, std::size_t cols, std::vector<T> data)
        {
            this->_data = Mat<T>(rows, cols);
            
            for(size_t i = 0; i < this->_data.n_rows; i++) {
                for(size_t j = 0; j < this->_data.n_cols; j++) {
                    this->_data(i,j) = data[i + j * this->_data.n_rows];
                }
            }
        }

        matrix(std::size_t rows, std::size_t cols)
        {
            this->_data = Mat<T>(rows, cols);
            this->_data.randn();
        }

        matrix(std::size_t rows, std::size_t cols, T pre)
        {
            this->_data = Mat<T>(rows, cols);
            
            for(size_t i = 0; i < this->_data.n_rows; i++) {
                for(size_t j = 0; j < this->_data.n_cols; j++) {
                    this->_data(i,j) = pre;
                }
            }
        }

        matrix(Mat<T> data)
        {
            auto y = exp(data);
            this->_data = data;
        }

        void copy_from(matrix<T, mode::cpu> &other)
        {
            this->_data = other._data;
        }

        matrix<T, mode::cpu> dot(matrix<T, mode::cpu> &other)
        {
            return matrix<T, mode::cpu>(this->_data * other._data);
        }

        matrix<T, mode::cpu>& maximum(T scalar)
        {
            // todo: implement me
            return *this;
        }

        static matrix<T, mode::cpu> maximum(matrix<T, mode::cpu> &other, T scalar)
        {
            // todo: implement me
            return other;
        }

        static matrix<T, mode::cpu> sign(matrix<T, mode::cpu> &other)
        {
            // todo: implement me
            return other;
        }

        std::size_t rows()
        {
            return this->_data.n_rows;
        }

        std::size_t cols()
        {
            return this->_data.n_cols;
        }

        bool operator==(const matrix<T, mode::cpu>& other)
        {
            if (&this->_data == &other._data)
                return true;
            if (this->_data.n_rows != other._data.n_rows)
                return false;
            if (this->_data.n_cols != other._data.n_cols)
                return false;
            
            for(size_t i = 0; i < this->_data.n_rows; i++) {
                for(size_t j = 0; j < this->_data.n_cols; j++) {
                    if(this->_data(i, j) != other._data(i, j)) {
                        return false;
                    }
                }
            }
            return true; 
        }
    };

    template<typename T>
    class matrix<T, mode::gpu> : public matrix_base<T, mode::gpu> {
    public:
        matrix(std::size_t rows, std::size_t cols, std::vector<T> data)
        {
            // todo: implement me
        }

        matrix(std::size_t rows, std::size_t cols)
        {
            // todo: implement me
        }


        std::size_t rows()
        {
            // todo: implement me
            return 0;
        }

        std::size_t cols()
        {
            // todo: implement me
            return 0;
        }

        void copy_from(matrix<T, mode::gpu> &other)
        {
            // todo: implement me
        }

        matrix<T, mode::gpu> dot(matrix<T, mode::gpu> &other)
        {
            // todo: implement me
            return other;
        }

        matrix<T, mode::gpu>& maximum(T scalar)
        {
            // todo: implement me
            return *this;
        }

        static matrix<T, mode::gpu> maximum(matrix<T, mode::gpu> &other, T scalar)
        {
            // todo: implement me
            return other;
        }

        static matrix<T, mode::gpu> sign(matrix<T, mode::gpu> &other)
        {
            // todo: implement me
            return other;
        }

        bool operator==(const matrix<T, mode::gpu>& other)
        {
            // todo: implement me
            return true;
        }
    };

    template class matrix_base<float, mode::cpu>;
    template class matrix_base<float, mode::gpu>;

    template class matrix_base<double, mode::cpu>;
    template class matrix_base<double, mode::gpu>;

    template class matrix<float, mode::cpu>;
    template class matrix<float, mode::gpu>;

    template class matrix<double, mode::cpu>;
    template class matrix<double, mode::gpu>;
}