#ifndef Dense_h
#define Dense_h

#include "utilities.h"
#include "layer.h"
#include "activation.h"
#include "opencl/dot.h"

namespace mozart
{
    template<typename T>
    class dense_config;

    template<typename T>
    class dense : public layer<T>
    {
        friend class dense_config<T>;
    public:
        activation<T> forward(matrix<T> &data);
        activation<T> train_forward(matrix<T> &data);
        void update_weights(matrix<T>& deltas);
        void update_bias(matrix<T>& deltas);
        virtual matrix<T>& weights();
    private:
        typename activation<T>::function _fun;
        matrix<T> _weights;
        matrix<T> _biases;
    };
}

#endif
