#ifndef InplaceReduceColumnSum_h
#define InplaceReduceColumnSum_h

#include "matrix.h"
#include "kernel.h"

namespace mozart
{
    template<typename T>
    class matrix;

    namespace function
    {
        template<typename T>
        void inplace_reduce_column_sum(matrix<T>& inout);
    }
}

#endif
