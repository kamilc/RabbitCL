#ifndef Input_h
#define Input_h

#include "matrix.h"
#include "utilities.h"
#include "layer.h"
#include "activation.h"

namespace mozart
{
    template<typename T>
    class input_config;

    template<typename T>
    class input : public layer<T>
    {
        friend class input_config<T>;
    public:
        activation<T> forward(matrix<T> &data);
        activation<T> train_forward(matrix<T> &data);
        void update_weights(matrix<T>& deltas);
        void update_bias(matrix<T>& deltas);
        virtual matrix<T>& weights();
    };
}

#endif
