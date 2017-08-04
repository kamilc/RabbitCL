#include "dense.h"

namespace mozart
{
    template<typename T>
    matrix<T> dense<T>::forward(matrix<T> &data)
    {
        // // first get the computed data from layers below:
        // auto in = this->_input->forward(data);

        // // next multiply in place by the weights:
        // auto out = in.dot(*(this->_weights));

        // normally hold this matrix as it will be needed but for now return it:

        // todo: implement me
        return data;
    }

    INSTANTIATE(dense);
}

