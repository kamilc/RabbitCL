#ifndef Dense_h
#define Dense_h

#include "layer.h"
#include "activation_function.h"

namespace heed
{
    template<typename T, mode MODE>
    class dense : public layer<T, MODE>
    {
    public:
        dense(std::size_t size, layer<T, MODE> &input, activation_function<T, MODE> fun);

        void forward(matrix<T, MODE> &data, matrix<T, MODE> &out);
    private:
        activation_function<T, MODE> &_nonlinearity;
    };
}

#endif