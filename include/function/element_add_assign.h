#ifndef ElementAddAssign_h
#define ElementAddAssign_h

#include "matrix.h"
#include "kernel.h"

namespace mozart
{
    template<typename T>
    class matrix;

    namespace function
    {
        template<typename T>
        void element_add_assign(const matrix<T>& lhs, const matrix<T>& rhs);
    }
}

#endif
