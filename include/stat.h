#ifndef Stat_h
#define Stat_h

#include "matrix.h"

using namespace mozart;

namespace mozart
{
    template<typename T>
    class stat
    {
    public:
        typedef stat<T> (*function)(matrix<T>&, matrix<T>&);
    };
}

#endif
