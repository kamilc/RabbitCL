#ifndef Squashmax_h
#define Squashmax_h

#include "matrix.h"
#include "utilities.h"
#include "kernel.h"

using namespace std;

namespace mozart
{
    namespace opencl
    {
        template<typename T>
        matrix<T> squashmax(matrix<T>& in);
    }
}

#endif

