#ifndef Function_h
#define Function_h

#include <functional>
#include <tuple>
#include "matrix.h"
#include "utilities.h"

using namespace std;

namespace mozart {

    template <typename T>
    class activation
    {
    public:
        typedef activation<T> (*function)(matrix<T>&, bool);

        matrix<T> out;
        matrix<T> deriv;

        activation(matrix<T> &in, bool derive);
        activation();
    };
}

#endif