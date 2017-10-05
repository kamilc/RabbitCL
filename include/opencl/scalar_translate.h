#ifndef ScalarTranslate_h
#define ScalarTranslate_h

#include "matrix.h"
#include "kernel.h"

namespace mozart
{
    template<typename T>
    class matrix;

    namespace opencl
    {
        template<typename T>
        matrix<T> scalar_translate(const matrix<T>& in, const T factor);
    }
}

#endif
