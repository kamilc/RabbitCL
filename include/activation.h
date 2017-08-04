#ifndef Function_h
#define Function_h

#include <functional>
#include <tuple>
#include "boost/optional.hpp"
#include "viennacl/matrix.hpp"
#include "utilities.h"

using namespace std;
using namespace boost;
using namespace viennacl;

namespace mozart {

    template <typename T>
    class activation
    {
    public:
        typedef activation<T> (*function)(matrix<T>&, bool);

        matrix<T> out;
        optional<matrix<T>> deriv;

        activation(matrix<T> &in, bool derive);
    };
}

#endif