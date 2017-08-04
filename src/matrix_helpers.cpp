#include "matrix_helpers.h"

namespace mozart {
    template<typename T>
    matrix<T> make_matrix(initializer_list<initializer_list<T>> values)
    {
        auto rows = 0;
        auto cols = 0;

        for(auto r : values) {
            ++rows;
            for(auto c: r) {
                ++cols;
            }
        }

        auto mat = matrix<T>(rows, cols);

        size_t rindex = 0;
        size_t cindex = 0;
        for(auto r  = values.begin(); r < values.end(); r++) {
            for(auto c = (*r).begin(); c < (*r).end(); c++) {
                mat(rindex, cindex) = *c;
            }
            ++rindex;
            ++cindex;
        }
        
        return mat;
    }

    template matrix<float> make_matrix(initializer_list<initializer_list<float>> values);
    template matrix<double> make_matrix(initializer_list<initializer_list<double>> values);
}
