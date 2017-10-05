#ifndef ActivationReLU_h
#define ActivationReLU_h

#include <boost/compute/container.hpp>
#include "matrix.h"
#include "utilities.h"
#include "activation.h"
#include "kernel.h"

using namespace std;

namespace mozart
{
    namespace opencl
    {
        template<typename T>
        activation<T> relu(matrix<T>& in, bool derive);
    }
}

#endif
