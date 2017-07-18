#ifndef Activation_h
#define Activation_h

#include "layer.h"
#include "activation_function.h"

namespace heed
{
    template<typename T, typename A>
    class activation : public layer<T>
    {
    private:
        A _func = {};
    public:
        activation(std::shared_ptr<layer<T>> input);

        std::shared_ptr<matrix<T>> forward(std::shared_ptr<matrix<T>> data);
    };
}


#endif