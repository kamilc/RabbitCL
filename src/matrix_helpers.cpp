#include "matrix_helpers.h"

namespace mozart {
    template<typename T>
    matrix<T> make_matrix(initializer_list<initializer_list<T>> values)
    {
        // todo: implement me
        return matrix<T>(0, 0);
    }

    template matrix<float> make_matrix(initializer_list<initializer_list<float>> values);
    template matrix<double> make_matrix(initializer_list<initializer_list<double>> values);
}
