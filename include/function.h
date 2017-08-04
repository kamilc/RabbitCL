#ifndef Function_h
#define Function_h

#include <functional>
#include <tuple>
#include "boost/optional.hpp"
#include "viennacl/matrix.hpp"

using namespace std;
using namespace boost;
using namespace viennacl;

namespace mozart {

    template <typename T>
    struct activation
    {
        typedef tuple<matrix<T>, optional<matrix<T>>> (*function)(matrix<T>&, bool);
    };
}

#endif