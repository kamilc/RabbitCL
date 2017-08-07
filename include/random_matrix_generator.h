#ifndef RandomMatrixGenerator_h
#define RandomMatrixGenerator_h

#include <random>
#include <vector>
#include "utilities.h"

using namespace std;

namespace mozart {
    template<typename T>
    class random_matrix_generator
    {
    private:
        std::random_device _rd;
        std::mt19937 _gen;
        std::normal_distribution<T> _dist;
        std::size_t _rows;
        std::size_t _cols;
        std::vector<T> _data;
    public:
        random_matrix_generator(std::size_t rows, std::size_t cols, T mean, T variance);
        random_matrix_generator(const random_matrix_generator &to_copy);

        T operator()(std::size_t row, std::size_t col) const;

        std::size_t size1() const;
        std::size_t size2() const;
    };
}

#endif