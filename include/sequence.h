#ifndef Sequence_h
#define Sequence_h

#include "viennacl/matrix.hpp"
#include "utilities.h"
#include "layer_config.h"

using namespace viennacl;

namespace heed
{
    template<typename T>
    class sequence
    {
    public:
        sequence& add(const layer_config<T> &config);

        virtual matrix<T> forward(matrix<T> &data);
    };
}

#endif