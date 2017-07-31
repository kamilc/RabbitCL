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
            std::copy(data.begin(), data.end(), this->_data.data().begin());
        }

        static matrix<T, mode::cpu> generate(std::size_t rows, std::size_t cols)
        {
            static std::normal_distribution<T> distribution(0, 1);
            static std::default_random_engine generator;

            std::vector<T> data = std::vector<T>(rows * cols);
            std::generate(data.begin(), data.end(), []() { return distribution(generator); });

            return matrix<T, mode::cpu>(rows, cols, data);
        }

        bool operator==(const matrix<T, mode::cpu>& other)
        {
            if (&this->_data == &other._data) 
                return true; 
            if (this->_data.size1() != other._data.size1()) 
                return false; 
            if (this->_data.size2() != other._data.size2()) 
                return false; 
            typename boost::numeric::ublas::matrix<T>::iterator1 l(this->_data.begin1()); 
            typename boost::numeric::ublas::matrix<T>::const_iterator1 r(other._data.begin1()); 
            while (l != this->_data.end1()) { 
                if (*l != *r) 
                    return false; 
                ++l; 
                ++r; 
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

        static matrix<T, mode::gpu> generate(std::size_t rows, std::size_t cols)
        {
            // todo: implement me

            return matrix<T, mode::gpu>(rows, cols, {});
        }

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