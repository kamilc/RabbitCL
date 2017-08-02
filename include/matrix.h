#ifndef Matrix_h
#define Matrix_h

#include <viennacl/matrix.hpp>
#include <eigen3/Eigen/Dense>

#include <boost/variant.hpp>
#include <random>
#include <type_traits>

using namespace Eigen;

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
        std::conditional_t<MODE == mode::cpu, Matrix<T, Dynamic, Dynamic>, viennacl::matrix<T>> _data;
    public:
        matrix(std::size_t rows, std::size_t cols, std::vector<T> data);
        matrix(std::size_t rows, std::size_t cols);
        matrix(std::size_t rows, std::size_t cols, T pre);

        std::size_t rows();
        std::size_t cols();

        void copy_from(matrix<T, MODE> &other);

        matrix<T, MODE> dot(matrix<T, MODE> &other);
        matrix<T, MODE>& maximum(T scalar);

        static matrix<T, MODE> maximum(matrix<T, MODE> &other, T scalar);
        static matrix<T, MODE> sign(matrix<T, MODE> &other);

        bool operator==(const matrix<T, MODE>& other);
    };
}

#endif