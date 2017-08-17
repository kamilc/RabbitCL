#ifndef Cost_h
#define Cost_h

#include <functional>
#include <tuple>
#include "boost/optional.hpp"
#include "viennacl/matrix.hpp"
#include "utilities.h"
#include "function/reduce_avg.h"

using namespace std;
using namespace boost;
using namespace viennacl;
using namespace mozart::function;

namespace mozart {

    template <typename T>
    class cost
    {
    public:
        typedef cost<T> (*function)(matrix<T>&, bool);

        matrix<T> out;
        matrix<T> deriv;

        cost(matrix<T> &in, bool derive);

        T avg();
    };
}

#endif