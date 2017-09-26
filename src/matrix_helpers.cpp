#include "matrix_helpers.h"

namespace mozart {
    template<typename T>
    matrix<T> make_matrix(initializer_list<initializer_list<T>> values)
    {
        auto rows = 0;
        auto cols = 0;

        for(auto r : values) {
            if(rows == 0) {
                for(auto c: r) {
                    cols++;
                }
            }
            rows++;
        }

        std::vector<T> data(rows * cols);
        auto mat = matrix<T>(rows, cols);

        size_t rindex = 0;
        for(auto r  = values.begin(); r < values.end(); r++) {
            size_t cindex = 0;
            for(auto c = (*r).begin(); c < (*r).end(); c++) {
                data[rindex * cols + cindex] = *c;
                cindex++;
            }
            rindex++;
        }

        mat.set_data(data);

        return mat;
    }

    template<typename T>
    matrix<T> make_random_matrix(std::size_t rows, std::size_t cols, T mean, T variance)
    {
        auto out = matrix<T>(rows, cols);

        out.fill_randn(mean, variance);

        return out;
    }

    template matrix<float> make_matrix(initializer_list<initializer_list<float>> values);
    template matrix<double> make_matrix(initializer_list<initializer_list<double>> values);

    template matrix<float> make_random_matrix(std::size_t rows, std::size_t cols, float mean, float variance);
    template matrix<double> make_random_matrix(std::size_t rows, std::size_t cols, double mean, double variance);
}
