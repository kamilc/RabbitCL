#ifndef CostSquaredError_h
#define CostSquaredError_h

#include "matrix.h"
#include "utilities.h"
#include "cost.h"
#include "kernel.h"

using namespace std;

namespace mozart
{
    namespace function
    {
        template<typename T>
        cost<T> squared_error(matrix<T>& in, matrix<T>& targets, bool derive);
    }
}

#endif