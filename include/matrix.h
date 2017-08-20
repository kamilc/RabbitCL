#ifndef Matrix_h
#define Matrix_h

#include <cstddef>
#include <boost/compute.hpp>
#include "utilities.h"

using namespace std;
using namespace boost;

namespace mozart
{
    template<typename T>
    class matrix
    {
    public:
        matrix();
        matrix(size_t size1, size_t size2);

        size_t size1();
        size_t size2();

        void set(size_t at1, size_t at2, T value);

        T operator()(size_t at1, size_t at2);
    private:
        bool transposed = false;
        compute::context _context;
        compute::vector<T> _data;
    };
}

#endif