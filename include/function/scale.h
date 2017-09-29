#ifndef Scale_h
#define Scale_h

#include "matrix.h"
#include "kernel.h"

namespace mozart
{
    template<typename T>
    class matrix;

    namespace function
    {
        template<typename T>
        matrix<T> scale(const matrix<T>& in, const T factor);
    }
}

#endif
