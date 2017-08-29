#ifndef Layer_h
#define Layer_h

#include <stdio.h>
#include "matrix.h"
#include "utilities.h"

using namespace std;

namespace mozart
{
    template<typename T>
    class activation;

    template<typename T>
    class layer
    {
    protected:
        size_t _size;
    public:
        size_t size();

        virtual activation<T> forward(matrix<T> &data) = 0;
        virtual activation<T> train_forward(matrix<T> &data) = 0;
        virtual void update_weights(matrix<T>& deltas) = 0;
    };
}

#endif
