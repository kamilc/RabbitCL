#ifndef Input_h
#define Input_h

#include "viennacl/matrix.hpp"
#include "utilities.h"
#include "layer.h"
#include "input_config.h"

using namespace viennacl;

namespace mozart 
{
    template<typename T>
    class input : public layer<T>
    {
    public:
        matrix<T> forward(matrix<T> &data);
    };
}

#endif