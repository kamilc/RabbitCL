#ifndef Function_h
#define Function_h

#include <functional>
#include <tuple>
#include "boost/optional.hpp"
#include "matrix.h"

using namespace std;
using namespace boost;

namespace heed {

    template <typename T, mode MODE>
    struct activation
    {
        typedef tuple<matrix<T, MODE>, optional<matrix<T, MODE>>> (*function)(matrix<T, MODE>&, bool);
    };
}

#endif