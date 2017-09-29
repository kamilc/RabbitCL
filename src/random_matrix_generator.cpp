#include "random_matrix_generator.h"

namespace mozart {
    template<typename T>
    random_matrix_generator<T>::random_matrix_generator(std::size_t rows, std::size_t cols, T mean, T variance)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<T> dist(mean, variance);

        this->_rows = rows;
        this->_cols = cols;
        this->_data = vector<T>(rows * cols);

        for(auto i = 0; i < rows*cols; i++) {
            this->_data[i] = dist(gen);
        }
    }

    template<typename T>
    random_matrix_generator<T>::random_matrix_generator(const random_matrix_generator &to_copy)
    {
        this->_rows = to_copy._rows;
        this->_cols = to_copy._cols;
        this->_data = to_copy._data;
    }

    template<typename T>
    std::size_t random_matrix_generator<T>::size1() const
    {
        return this->_rows;
    }

    template<typename T>
    std::size_t random_matrix_generator<T>::size2() const
    {
        return this->_cols;
    }

    template<typename T>
    T random_matrix_generator<T>::operator()(std::size_t row, std::size_t col) const
    {
        return this->_data[row*this->_cols + col];
    }

    INSTANTIATE(random_matrix_generator);
}
