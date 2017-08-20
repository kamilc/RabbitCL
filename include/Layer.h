#ifndef Layer_h
#define Layer_h

#include <stdio.h>
#include <boost/optional.hpp>
#include "matrix.h"

#include "utilities.h"


using namespace std;
using namespace boost;

namespace mozart
{
    template<typename T>
    class layer
    {
    protected:
        size_t _size;
    public:
        size_t size();

        virtual matrix<T> forward(matrix<T> &data) = 0;
        virtual void update_weights(matrix<T>& deltas) = 0;
    };
}

#endif
