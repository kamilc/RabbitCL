#ifndef InplaceColumnwiseAdd_h
#define InplaceColumnwiseAdd_h

#include "matrix.h"
#include "kernel.h"

namespace mozart
{
    template<typename T>
    class matrix;

    namespace opencl
    {
        template<typename T>
        void inplace_columnwise_subtract(matrix<T>& left, const matrix<T>& right);
    }
}

#endif
