#include "activation.h"

namespace mozart {
    template<typename T>
    activation<T>::activation(matrix<T> &in, bool derive)
    {
        this->out = matrix<T>(in.size1(), in.size2());
        if(derive){
            this->deriv = matrix<T>(in.size1(), in.size2());
        }
    }

    INSTANTIATE(activation);
}