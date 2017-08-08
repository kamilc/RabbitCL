#ifndef Softmax_h
#define Softmax_h

#include <tuple>
#include "boost/optional.hpp"
#include "viennacl/matrix.hpp"
#include "utilities.h"
#include "activation.h"

using namespace std;
using namespace boost;
using namespace viennacl;

namespace mozart
{
    namespace function
    {
        template<typename T>
        activation<T> softmax(matrix<T>& in, bool derive);
    }
}

#endif