#include "cost.h"

namespace mozart {

    template<typename T>
    cost<T>::cost(matrix<T> &in, bool derive)
    {
        this->out = 0;
        if(derive){
            this->deriv = matrix<T>(in.size1(), in.size2());
        }
        else {
            this->deriv = matrix<T>(0, 0);
        }
    }

    INSTANTIATE(cost);
}