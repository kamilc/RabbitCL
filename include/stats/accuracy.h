#ifndef Accuracy_h
#define Accuracy_h

#include "matrix.h"
#include "stat.h"
#include "opencl/squashmax.h"

using namespace mozart::opencl;

namespace mozart
{
    namespace stats
    {
        template<typename T>
        stat<T> accuracy(matrix<T>&, matrix<T>&);
    }
}

#endif
