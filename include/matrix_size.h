#ifndef MatrixSize_h
#define MatrixSize_h

#include <cstddef>
#include <boost/compute/types/struct.hpp>
#include <boost/compute/type_traits/type_name.hpp>
#include <boost/compute/type_traits/type_definition.hpp>
#include <boost/compute/type_traits.hpp>

using namespace std;
using namespace boost;

namespace mozart
{
    struct matrix_size
    {
        unsigned int size1;
        unsigned int size2;
        unsigned int start1;
        unsigned int start2;
        unsigned int internal_size1;
        unsigned int internal_size2;
    };
}

#endif
