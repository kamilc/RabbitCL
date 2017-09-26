#ifndef InplaceColumnwiseAdd_h
#define InplaceColumnwiseAdd_h

#include "matrix.h"
#include "kernel.h"

namespace mozart
{
    template<typename T>
    class matrix;

    namespace function
    {
        template<typename T>
        void inplace_columnwise_add(matrix<T>& left, const matrix<T>& right);
    }
}

#endif
