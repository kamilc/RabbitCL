#include "matrix.h"

using namespace Eigen;

namespace heed
{
    template<typename T>
    class matrix<T, mode::cpu> : public matrix_base<T, mode::cpu> {
    private:
        Matrix<T, Dynamic, Dynamic> _data;
    public:
        matrix(std::size_t rows, std::size_t cols, std::vector<T> data)
        {
            this->_data = Matrix<T, Dynamic, Dynamic>(rows, cols);
            
            for(size_t i = 0; i < this->_data.rows(); i++) {
                for(size_t j = 0; j < this->_data.cols(); j++) {
                    this->_data(i,j) = data[i + j * this->_data.rows()];
                }
            }
        }

        matrix(std::size_t rows, std::size_t cols)
        {
            this->_data = Matrix<T, Dynamic, Dynamic>::Random(rows, cols);
        }

        matrix(std::size_t rows, std::size_t cols, T pre)
        {
            this->_data = Matrix<T, Dynamic, Dynamic>(rows, cols);
            
            for(size_t i = 0; i < this->_data.rows(); i++) {
                for(size_t j = 0; j < this->_data.cols(); j++) {
                    this->_data(i,j) = pre;
                }
            }
        }

        matrix(Matrix<T, Dynamic, Dynamic> data)
        {
            this->_data = data;
        }

        void copy_from(matrix<T, mode::cpu> &other)
        {
            this->_data = other._data;
        }

        matrix<T, mode::cpu> dot(matrix<T, mode::cpu> &other)
        {
            // std::cout << "LHS: " << this->_data << std::endl;
            // std::cout << "RHS: " << other._data << std::endl;

            return matrix<T, mode::cpu>(this->_data * other._data);
        }

        std::size_t rows()
        {
            return this->_data.rows();
        }

        std::size_t cols()
        {
            return this->_data.cols();
        }

        bool operator==(const matrix<T, mode::cpu>& other)
        {
            if (&this->_data == &other._data)
                return true;
            if (this->_data.rows() != other._data.rows())
                return false;
            if (this->_data.cols() != other._data.cols()) 
                return false;
            
            for(size_t i = 0; i < this->_data.rows(); i++) {
                for(size_t j = 0; j < this->_data.cols(); j++) {
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

        bool operator==(const matrix<T, mode::cpu>& other)
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