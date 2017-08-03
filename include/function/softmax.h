#ifndef Softmax_h
#define Softmax_h

#include "utilities.h"
#include "matrix.h"

#include <tuple>
#include "boost/optional.hpp"

using namespace std;
using namespace boost;

namespace heed
{
    namespace function
    {
        template<typename T, mode MODE>
        tuple<matrix<T, MODE>, optional<matrix<T, MODE>>> softmax(matrix<T, MODE>& in, bool derive)
        {
            // todo: implement me
            return in;
        }
    }
}

#endif