#ifndef Matrix_h
#define Matrix_h

#include <viennacl/matrix.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/variant.hpp>
#include <random>

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
    public:
        matrix(std::size_t rows, std::size_t cols, std::vector<T> data);
        matrix(std::size_t rows, std::size_t cols);
        matrix(std::size_t rows, std::size_t cols, T pre);

        std::size_t rows();
        std::size_t cols();

        //static matrix generate(std::tuple<std::size_t, std::size_t> *size);
        bool operator==(const matrix<T, MODE>& other);
    };
}

#endif