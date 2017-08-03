#ifndef ActivationReLU_h
#define ActivationReLU_h

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
        // template<typename T, mode MODE>
        // class relu : public activation_function<T, MODE>
        // {
        // public:
        //     matrix<T, MODE> compute(matrix<T, MODE> &data);
        //     matrix<T, MODE> derivation_slope(matrix<T, MODE> &data);
        // };

        template<typename T, mode MODE>
        tuple<matrix<T, MODE>, optional<matrix<T, MODE>>> relu(matrix<T, MODE>& in)
        {
            // todo: implement me
            return in;
        }
    }
}

#endif