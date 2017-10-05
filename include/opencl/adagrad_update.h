#ifndef AdagradUpdate_h
#define AdagradUpdate_h

#include "matrix.h"

namespace mozart
{
    namespace opencl
    {
        template<typename T>
        void adagrad_update(T alpha, matrix<T>& weight_delta, matrix<T>& memo, T eps);
    }
}

#endif
