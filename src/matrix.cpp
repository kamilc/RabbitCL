#include "matrix.h"

namespace heed
{
    template<typename T, mode MODE>
    matrix_base<T, MODE>::matrix_base(std::size_t rows, std::size_t cols, std::vector<T> data)
    {
        // todo: implement me
    }

    template<typename T, mode MODE>
    matrix_base<T, MODE>::matrix_base(viennacl::matrix<T> data)
    {
        // todo: implement me
    }

    template<typename T, mode MODE>
    matrix_base<T, MODE>::matrix_base(boost::numeric::ublas::matrix<T> data)
    {
        // todo: implement me
    }

    template<typename T>
    class matrix<T, mode::cpu> : public matrix_base<T, mode::cpu> {
    public:
        matrix(boost::numeric::ublas::matrix<T> data) : matrix_base<T, mode::cpu>(data) {}

        matrix(viennacl::matrix<T> data) : matrix_base<T, mode::cpu>(data) {}

        matrix(std::size_t rows, std::size_t cols, std::vector<T> data) : matrix_base<T, mode::cpu>(rows, cols, data) {}

        static matrix<T, mode::cpu> generate(std::size_t rows, std::size_t cols)
        {
            // todo: implement me

            return matrix<T, mode::cpu>(rows, cols, {});
        }

        bool operator==(const matrix<T, mode::cpu>& other)
        {
            return true;
        }
    };

    template<typename T>
    class matrix<T, mode::gpu> : public matrix_base<T, mode::gpu> {
    public:
        matrix(boost::numeric::ublas::matrix<T> data) : matrix_base<T, mode::gpu>(data) {}

        matrix(viennacl::matrix<T> data) : matrix_base<T, mode::gpu>(data) {}

        matrix(std::size_t rows, std::size_t cols, std::vector<T> data) : matrix_base<T, mode::gpu>(rows, cols, data) {}

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


    // template<typename T, mode MODE>
    // bool matrix_base<T, MODE>::operator==(const matrix<T, MODE> &other)
    // {
    //     // todo: implement me
    //     return true;
    // }

    // template<typename T>
    // matrix<T, mode::cpu> matrix<T, mode::cpu>::generate(std::size_t rows, std::size_t cols)
    // {
    //     // todo: implement me

    //     return matrix<T, mode::cpu>(rows, cols, {});
    // }

    // template<typename T>
    // matrix<T, mode::gpu> matrix<T, mode::gpu>::generate(std::size_t rows, std::size_t cols)
    // {
    //     // todo: implement me

    //     return matrix<T, mode::gpu>(rows, cols, {});
    // }

    template class matrix_base<float, mode::cpu>;
    template class matrix_base<float, mode::gpu>;

    template class matrix_base<double, mode::cpu>;
    template class matrix_base<double, mode::gpu>;

    template class matrix<float, mode::cpu>;
    template class matrix<float, mode::gpu>;

    template class matrix<double, mode::cpu>;
    template class matrix<double, mode::gpu>;
}