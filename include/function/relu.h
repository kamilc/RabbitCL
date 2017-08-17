#ifndef ActivationReLU_h
#define ActivationReLU_h

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
        activation<T> relu(matrix<T>& in, bool derive);

        KERNEL_CLASS(relu_kernel);
        KERNEL_CLASS(relu_deriv_kernel);
    }
}

#endif