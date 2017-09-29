#include "activation.h"

namespace mozart {
    template<typename T>
    activation<T>::activation(matrix<T> &in, bool derive)
    {
        this->out = matrix<T>(in.size1(), in.size2());
        if(derive){
            this->deriv = matrix<T>(in.size1(), in.size2());
        }
        else {
            this->deriv = matrix<T>(0, 0);
        }
    }

    template<typename T>
    activation<T>::activation()
    {
        // no-op
    }

    template<typename T>
    activation<T> activation<T>::with(matrix<T>& data)
    {
        activation<T> out(data, false);

        out.out = data;

        return out;
    }

    INSTANTIATE(activation);
}
