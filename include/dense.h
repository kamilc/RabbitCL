#ifndef Dense_h
#define Dense_h

#include "utilities.h"
#include "layer.h"
#include "dense_config.h"
#include "activation.h"

namespace mozart
{
    template<typename T>
    class dense : public layer<T>
    {
    public:
        matrix<T> forward(matrix<T> &data);
    private:
        typename activation<T>::function _fun;
    };
}

#endif