#ifndef ReduceAvg_h
#define ReduceAvg_h

#include <boost/compute/container.hpp>
#include "matrix.h"
#include "scalar.h"
#include "utilities.h"
#include "local.h"
#include "kernel.h"

using namespace std;

namespace mozart
{
    namespace function
    {
        template<typename T>
        scalar<T> reduce_avg(matrix<T>& in);
    }
}

#endif
