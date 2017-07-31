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
        dense(std::size_t size, std::shared_ptr<layer<T, MODE>> input, activation_function<T, MODE> fun);

        static std::shared_ptr<dense<T, MODE>> define(std::size_t size, std::shared_ptr<layer<T, MODE>> input, activation_function<T, MODE> fun);

        std::shared_ptr<matrix<T, MODE>> forward(std::shared_ptr<matrix<T, MODE>> data);
    private:
        activation_function<T, MODE> &_nonlinearity;
    };
}

#endif