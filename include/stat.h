#ifndef Stat_h
#define Stat_h

#include "matrix.h"
#include <string>

using namespace std;
using namespace mozart;

namespace mozart
{
    template<typename T>
    class stat
    {
    public:
        typedef stat<T> (*function)(matrix<T>&, matrix<T>&);

        T out;
        size_t count;
        string name;
    };
}

#endif
