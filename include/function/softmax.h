#ifndef Softmax_h
#define Softmax_h

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
        tuple<matrix<T>, optional<matrix<T>>> softmax(matrix<T>& in, bool derive)
        {
            // todo: implement me
            return in;
        }
    }
}

#endif