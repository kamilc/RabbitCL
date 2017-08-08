#ifndef Softmax_h
#define Softmax_h

#include <tuple>
#include "boost/optional.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/ocl/local_mem.hpp"
#include "utilities.h"
#include "activation.h"

using namespace std;
using namespace boost;
using namespace viennacl;
using namespace viennacl::ocl;

namespace mozart
{
    namespace function
    {
        template<typename T>
        activation<T> softmax(matrix<T>& in, bool derive);
    }
}

#endif