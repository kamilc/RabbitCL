#ifndef RMSPropUpdate_h
#define RMSPropUpdate_h

#include "matrix.h"

namespace mozart
{
    namespace opencl
    {
        template<typename T>
        void rmsprop_update(T alpha, matrix<T>& weight_delta, matrix<T>& memo, T eps, T mu);
    }
}

#endif
