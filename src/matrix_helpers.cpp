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

        auto mat = matrix<T>(rows, cols);

        size_t rindex = 0;
        size_t cindex = 0;
        for(auto r  = values.begin(); r < values.end(); r++) {
            cindex = 0;
            for(auto c = (*r).begin(); c < (*r).end(); c++) {
                mat(rindex, cindex) = *c;
                cindex++;
            }
            rindex++;
        }
        
        return mat;
    }

    template<typename T>
    matrix<T> make_random_matrix(std::size_t rows, std::size_t cols, T mean, T variance)
    {
        auto out = matrix<T, column_major>(rows, cols);
        auto generator = random_matrix_generator<T>(rows, cols, mean, variance);

        viennacl::copy(generator, out);

        return out;
    }

    template matrix<float> make_matrix(initializer_list<initializer_list<float>> values);
    template matrix<double> make_matrix(initializer_list<initializer_list<double>> values);

    template matrix<float> make_random_matrix(std::size_t rows, std::size_t cols, float mean, float variance);
    template matrix<double> make_random_matrix(std::size_t rows, std::size_t cols, double mean, double variance);
}
