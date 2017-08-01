#ifndef Matrix_h
#define Matrix_h

#include <viennacl/matrix.hpp>
#include <eigen3/Eigen/Dense>

#include <boost/variant.hpp>
#include <random>

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
        Matrix<T, Dynamic, Dynamic> _data;
    public:
        matrix(std::size_t rows, std::size_t cols, std::vector<T> data);
        matrix(std::size_t rows, std::size_t cols);
        matrix(std::size_t rows, std::size_t cols, T pre);

        std::size_t rows();
        std::size_t cols();

        void copy_from(matrix<T, MODE> &other);

        matrix<T, MODE> dot(matrix<T, MODE> &other);

        bool operator==(const matrix<T, MODE>& other);
    };
}

#endif