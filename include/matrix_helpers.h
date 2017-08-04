#ifndef MozartHelpers_h
#define MozartHelpers_h

#include "viennacl/matrix.hpp"
#include <initializer_list>

using namespace std;
using namespace viennacl;

namespace mozart {
    template<typename T>
    matrix<T> make_matrix(initializer_list<initializer_list<T>> values);
}

#endif