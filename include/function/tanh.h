#ifndef ActivationTanh_h
#define ActivationTanh_h

#include <boost/compute/container.hpp>
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
        activation<T> tanh(matrix<T>& in, bool derive);
    }
}

#endif
