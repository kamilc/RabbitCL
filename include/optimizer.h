#ifndef Optimizer_h
#define Optimizer_h

#include "viennacl/matrix.hpp"
#include "utilities.h"
#include "sequence.h"

using namespace viennacl;

namespace mozart
{
    template<typename T>
    class optimizer
    {
    public:
        virtual void run(sequence<T> &network, matrix<T> &data, matrix<T> &targets) = 0;
    };
}

#endif