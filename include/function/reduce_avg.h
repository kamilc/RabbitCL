#ifndef ReduceAvg_h
#define ReduceAvg_h

#include "viennacl/matrix.hpp"
#include "viennacl/scalar.hpp"
#include "viennacl/ocl/backend.hpp"
#include "viennacl/backend/memory.hpp"
#include "utilities.h"
#include "kernel_class.h"

using namespace std;
using namespace viennacl;
using namespace viennacl::backend;

namespace mozart
{
    namespace function
    {
        template<typename T>
        scalar<T> reduce_avg(matrix<T>& in);

        SCALAR_KERNEL_CLASS(reduce_avg_kernel);
    }
}

#endif