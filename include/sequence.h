#ifndef Sequence_h
#define Sequence_h

#include "utilities.h"
#include "matrix.h"
#include "layer_config.h"

namespace heed
{
    template<typename T, mode MODE>
    class sequence
    {
    public:
        sequence& add(const layer_config<T, MODE> &config);

        virtual matrix<T, MODE> forward(matrix<T, MODE> &data);
    };
}

#endif