#include "cost.h"

namespace mozart {

    template<typename T>
    cost<T>::cost(matrix<T> &in, bool derive)
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
    scalar<T> cost<T>::avg()
    {
        return reduce_avg<T>(this->out);
    }

    INSTANTIATE(cost);
}
