#ifndef Adagrad_h
#define Adagrad_h

#include "gradient_descent.h"
#include "matrix.h"
#include "utilities.h"
#include "sequence.h"

namespace mozart
{
    namespace optimizers
    {
        template<typename T>
        class adagrad : public gradient_descent<T>
        {
        };
    }
}

#endif
