#ifndef Input_h
#define Input_h

#include "matrix.h"
#include "utilities.h"
#include "layer.h"



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
        void update_weights(matrix<T>& deltas);
    };
}

#endif