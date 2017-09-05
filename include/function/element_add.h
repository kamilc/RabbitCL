#ifndef ElementAdd_h
#define ElementAdd_h

#include "matrix.h"
#include "kernel.h"

namespace mozart
{
    template<typename T>
    class matrix;

    namespace function
    {
        template<typename T>
        matrix<T> element_add(const matrix<T>& lhs, const matrix<T>& rhs);
    }
}

#endif