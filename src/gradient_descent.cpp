#include "gradient_descent.h"

namespace heed
{
    template<typename T, mode MODE>
    gradient_descent<T, MODE>::gradient_descent()
    {
        // no-op
    }

    template<typename T, mode MODE>
    void gradient_descent<T, MODE>::run(layer<T, MODE> &network)
    {
        // todo: implement me
    }

    template<typename T, mode MODE>
    gradient_descent<T, MODE>& gradient_descent<T, MODE>::setEpochs(unsigned long epochs)
    {
        // todo: implement me
        return *this;
    }

    template<typename T, mode MODE>
    gradient_descent<T, MODE>& gradient_descent<T, MODE>::setBatches(unsigned long batches)
    {
        // todo: implement me
        return *this;
    }

    template<typename T, mode MODE>
    gradient_descent<T, MODE>& gradient_descent<T, MODE>::setEta(T eta)
    {
        // todo: implement me
        return *this;
    }

    template class gradient_descent<float, mode::cpu>;
    template class gradient_descent<float, mode::gpu>;

    template class gradient_descent<double, mode::cpu>;
    template class gradient_descent<double, mode::gpu>;
}