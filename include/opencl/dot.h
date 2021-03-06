#ifndef Dot_h
#define Dot_h

// #include "clBLAS.h"
#include "clblast.h"
#include "matrix.h"
#include "kernel.h"

using namespace std;
using namespace mozart;

namespace mozart
{
    namespace opencl
    {
        template<typename T>
        matrix<T> dot(matrix<T>& lhs, matrix<T>& rhs, bool lhs_transpose = false, bool rhs_transpose = false);
    }
}

#endif
