#ifndef Softmax_h
#define Softmax_h

#include "viennacl/matrix.hpp"
#include "viennacl/ocl/local_mem.hpp"
#include "utilities.h"
#include "activation.h"
#include "kernel.h"

using namespace std;
using namespace viennacl;
using namespace viennacl::ocl;
using namespace viennacl::backend;

namespace mozart
{
    namespace function
    {
        template<typename T>
        activation<T> softmax(matrix<T>& in, bool derive);
    }
}

#endif