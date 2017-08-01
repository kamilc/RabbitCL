#include "matrix.h"

namespace heed
{
    template<typename T>
    class matrix<T, mode::cpu> : public matrix_base<T, mode::cpu> {
    private:
        boost::numeric::ublas::matrix<T> _data;
    public:
        matrix(std::size_t rows, std::size_t cols, std::vector<T> data)
        {
            this->_data = boost::numeric::ublas::matrix<T>(rows, cols);
            
            for(size_t i = 0; i < this->_data.size1(); i++) {
                for(size_t j = 0; j < this->_data.size2(); j++) {
                    this->_data(i,j) = data[i + j * this->_data.size1()];
                }
            }
        }

        matrix(std::size_t rows, std::size_t cols)
        {
            static std::normal_distribution<T> distribution(0, 1);
            static std::default_random_engine generator;

            auto data = std::vector<T>(rows * cols);
            
            std::generate(data.begin(), data.end(), []() { return distribution(generator); });

            this->_data = boost::numeric::ublas::matrix<T>(rows, cols);
            std::copy(data.begin(), data.end(), this->_data.data().begin());
        }

        matrix(std::size_t rows, std::size_t cols, T pre)
        {
            this->_data = boost::numeric::ublas::matrix<T>(rows, cols);
            
            for(size_t i = 0; i < this->_data.size1(); i++) {
                for(size_t j = 0; j < this->_data.size2(); j++) {
                    this->_data(i,j) = pre;
                }
            }
        }

        std::size_t rows()
        {
            return this->_data.size1();
        }

        std::size_t cols()
        {
            return this->_data.size2();
        }

        bool operator==(const matrix<T, mode::cpu>& other)
        {
            if (&this->_data == &other._data)
                return true;
            if (this->_data.size1() != other._data.size1())
                return false;
            if (this->_data.size2() != other._data.size2()) 
                return false;
            
            for(size_t i = 0; i < this->_data.size1(); i++) {
                for(size_t j = 0; j < this->_data.size2(); j++) {
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

        }

        matrix(std::size_t rows, std::size_t cols)
        {

        }


        std::size_t rows()
        {
            return 0;
        }

        std::size_t cols()
        {
            return 0;
        }

        // static matrix<T, mode::gpu> generate(std::tuple<std::size_t, std::size_t> *size)
        // {
        //     // todo: implement me

        //     auto rows = std::get<0>(*size);
        //     auto cols = std::get<1>(*size);

        //     return matrix<T, mode::gpu>(rows, cols, {});
        // }

        bool operator==(const matrix<T, mode::cpu>& other)
        {
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