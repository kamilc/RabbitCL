#ifndef ActivationReLU_h
#define ActivationReLU_h

#include <tuple>
#include "boost/optional.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/ocl/backend.hpp"
#include "utilities.h"
#include "activation.h"

using namespace std;
using namespace viennacl;
using namespace viennacl::linalg;
using namespace boost;

namespace mozart
{
    namespace function
    {
        template<typename T>
        activation<T> relu(matrix<T>& in, bool derive);
    }
}

#endif