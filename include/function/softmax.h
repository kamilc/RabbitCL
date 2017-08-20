#ifndef Softmax_h
#define Softmax_h

#include "matrix.h"

#include "utilities.h"
#include "activation.h"
#include "kernel.h"

using namespace std;




namespace mozart
{
    namespace function
    {
        template<typename T>
        activation<T> softmax(matrix<T>& in, bool derive);
    }
}

#endif