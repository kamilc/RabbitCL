#ifndef Dot_h
#define Dot_h

#include <clBLAS.h>
#include "matrix.h"
#include "kernel.h"

using namespace std;
using namespace mozart;

namespace mozart
{
    namespace function
    {
        template<typename T>
        matrix<T> dot(matrix<T>& lhs, matrix<T>& rhs);
    }
}

#endif