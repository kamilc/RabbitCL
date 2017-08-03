#ifndef Input_h
#define Input_h

#include "utilities.h"
#include "layer.h"
#include "input_config.h"

namespace heed 
{
    template<typename T, mode MODE>
    class input : public layer<T, MODE>
    {
    public:
        input(input_config<T, MODE> &config);

        static input_config<T, MODE> with(std::size_t size);

        matrix<T, MODE> forward(matrix<T, MODE> &data);
    };
}

#endif