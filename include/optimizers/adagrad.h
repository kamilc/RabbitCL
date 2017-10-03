#ifndef Adagrad_h
#define Adagrad_h

#include "optimizers/stateful.h"
#include "matrix.h"
#include "utilities.h"
#include "sequence.h"

namespace mozart
{
    namespace optimizers
    {
        template<typename T>
        class adagrad : stateful<T, 1>
        {
        };
    }
}

#endif
