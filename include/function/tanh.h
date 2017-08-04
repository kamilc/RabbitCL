#ifndef ActivationTanh_h
#define ActivationTanh_h

#include <tuple>
#include "boost/optional.hpp"
#include "viennacl/matrix.hpp"
#include "utilities.h"

using namespace std;
using namespace boost;
using namespace viennacl;

namespace heed
{
    namespace function
    {
        template<typename T>
        tuple<matrix<T>, optional<matrix<T>>> tanh(matrix<T>& in, bool derive)
        {
            // todo: implement me
            return in;
        }
    }
}

#endif