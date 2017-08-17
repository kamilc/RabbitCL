#ifndef ActivationTanh_h
#define ActivationTanh_h

#include "viennacl/matrix.hpp"
#include "viennacl/ocl/backend.hpp"
#include "viennacl/backend/memory.hpp"
#include "utilities.h"
#include "activation.h"
#include "kernel_class.h"

using namespace std;
using namespace viennacl;
using namespace viennacl::backend;

namespace mozart
{
    namespace function
    {
        template<typename T>
        activation<T> tanh(matrix<T>& in, bool derive);

        KERNEL_CLASS(tanh_kernel);
        KERNEL_CLASS(tanh_deriv_kernel);
    }
}

#endif