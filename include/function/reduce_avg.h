#ifndef ReduceAvg_h
#define ReduceAvg_h

#include "matrix.h"
#include "scalar.h"
#include "utilities.h"
#include "kernel.h"

using namespace std;

namespace mozart
{
    namespace function
    {
        template<typename T>
        scalar<T> reduce_avg(matrix<T>& in);

        KERNEL(reduce_avg_kernel);
    }
}

#endif