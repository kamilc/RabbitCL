#ifndef ActivationReLU_h
#define ActivationReLU_h

#include <tuple>
#include "boost/optional.hpp"
#include "viennacl/matrix.hpp"
#include "utilities.h"

using namespace std;
using namespace viennacl;
using namespace boost;

namespace mozart
{
    namespace function
    {
        template<typename T>
        tuple<matrix<T>, optional<matrix<T>>> relu(matrix<T>& in, bool derive)
        {
            // todo: implement me
            return in;
        }
    }
}

#endif