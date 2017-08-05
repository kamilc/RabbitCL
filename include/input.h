#ifndef Input_h
#define Input_h

#include "viennacl/matrix.hpp"
#include "utilities.h"
#include "layer.h"

using namespace viennacl;

namespace mozart 
{
    template<typename T>
    class input_config;

    template<typename T>
    class input : public layer<T>
    {
        friend class input_config<T>;
    public:
        matrix<T> forward(matrix<T> &data);
    };
}

#endif