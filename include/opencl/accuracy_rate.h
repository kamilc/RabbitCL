#ifndef AccuracyRate_h
#define AccuracyRate_h

#include "matrix.h"
#include "scalar.h"
#include "local.h"
#include "kernel.h"

namespace mozart
{
    namespace opencl
    {
        template<typename T>
        scalar<T> accuracy_rate(const matrix<T>& predicted, const matrix<T>& targets);
    }
}

#endif
