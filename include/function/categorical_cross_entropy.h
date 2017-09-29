#ifndef CostCatCrossEntropy_h
#define CostCatCrossEntropy_h

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
        cost<T> categorical_cross_entropy(matrix<T>& in, matrix<T>& targets, bool derive);
    }
}

#endif
