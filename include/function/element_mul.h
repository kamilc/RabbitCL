#ifndef ElementMul_h
#define ElementMul_h

#include "matrix.h"
#include "kernel.h"

namespace mozart
{
    template<typename T>
    class matrix;

    namespace function
    {
        template<typename T>
        matrix<T> element_mul(const matrix<T>& lhs, const matrix<T>& rhs);
    }
}

#endif