#ifndef Cost_h
#define Cost_h

#include <functional>
#include <tuple>
#include "boost/optional.hpp"
#include "matrix.h"
#include "scalar.h"
#include "utilities.h"
#include "opencl/reduce_avg.h"

using namespace std;
using namespace boost;

using namespace mozart::opencl;

namespace mozart {

    template <typename T>
    class cost
    {
    public:
        typedef cost<T> (*function)(matrix<T>&, matrix<T>&, bool);

        matrix<T> out;
        matrix<T> deriv;

        cost(matrix<T> &in, bool derive);

        scalar<T> avg();
    };
}

#endif
