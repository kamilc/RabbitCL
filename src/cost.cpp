#include "cost.h"

namespace mozart {

    template<typename T>
    cost<T>::cost(matrix<T> &in, bool derive)
    {
        this->out = matrix<T>(in.size1(), 1);
        if(derive){
            this->deriv = matrix<T>(in.size1(), 1);
        }
        else {
            this->deriv = matrix<T>(0, 0);
        }
    }

    template<typename T>
    T cost<T>::avg()
    {
        return reduce_avg<T>(this->out);
    }

    INSTANTIATE(cost);
}