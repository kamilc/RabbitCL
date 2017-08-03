#ifndef Matrix_h
#define Matrix_h

#include <viennacl/matrix.hpp>
#include <boost/variant.hpp>
#include <random>
#include <type_traits>
#include <armadillo>

#include "utilities.h"

using namespace arma;

namespace heed {

    enum mode
    {
        cpu,
        gpu
    };

    template<typename T, mode MODE>
    class matrix_base{
    };

    template<typename T, mode MODE>
    class matrix : public matrix_base<T, MODE> {
    private:
        std::conditional_t<MODE == mode::cpu, Mat<T>, viennacl::matrix<T>> _data;
    public:
        matrix(std::size_t rows, std::size_t cols, std::vector<T> data);
        matrix(std::size_t rows, std::size_t cols);
        matrix(std::size_t rows, std::size_t cols, T pre);

        std::size_t rows();
        std::size_t cols();

        void copy_from(matrix<T, MODE> &other);

        matrix<T, MODE> dot(matrix<T, MODE> &other);
        matrix<T, MODE>& maximum(T scalar);
        matrix<T, MODE>& maximum();
        matrix<T, MODE>& slice_rows(std::size_t start, std::size_t end);

        T sum();

        static matrix<T, MODE> maximum(matrix<T, MODE> &other, T scalar);
        static matrix<T, MODE> sign(matrix<T, MODE> &other);
        static matrix<T, MODE> exp(matrix<T, MODE> &other);

        bool operator==(const matrix<T, MODE>& other);
        matrix<T, MODE> operator-(const matrix<T, MODE>& other);
        matrix<T, MODE> operator/(const T scalar);
    };

    template<typename T, mode MODE>
    matrix<T, MODE> operator-(T scalar, const matrix<T, MODE>& other);
}

#endif