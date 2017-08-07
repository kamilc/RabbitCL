#ifndef MozartHelpers_h
#define MozartHelpers_h

#include "viennacl/matrix.hpp"
#include <initializer_list>
#include <random>
#include "random_matrix_generator.h"

using namespace std;
using namespace viennacl;

namespace mozart {
    template<typename T>
    matrix<T> make_matrix(initializer_list<initializer_list<T>> values);

    template<typename T>
    matrix<T> make_random_matrix(std::size_t rows, std::size_t cols, T mean, T variance);
}

#endif