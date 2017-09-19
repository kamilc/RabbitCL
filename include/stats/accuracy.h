#ifndef Accuracy_h
#define Accuracy_h

#include "matrix.h"
#include "stat.h"

namespace mozart
{
    namespace stats
    {
        template<typename T>
        stat<T> accuracy(matrix<T>&, matrix<T>&);
    }
}

#endif
