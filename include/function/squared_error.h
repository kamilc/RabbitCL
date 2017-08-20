#ifndef CostSquaredError_h
#define CostSquaredError_h

#include "viennacl/matrix.hpp"
#include "viennacl/ocl/backend.hpp"
#include "viennacl/backend/memory.hpp"
#include "utilities.h"
#include "cost.h"
#include "kernel.h"

using namespace std;
using namespace viennacl;
using namespace viennacl::backend;

namespace mozart
{
    namespace function
    {
        template<typename T>
        cost<T> squared_error(matrix<T>& in, matrix_base<T>& targets, bool derive);
    }
}

#endif