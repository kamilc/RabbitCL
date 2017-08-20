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
        matrix(size_t size1, size_t size2, compute::context context);

        static compute::context default_context();
        static compute::command_queue default_queue();

        size_t size1();
        size_t size2();

        void set(size_t at1, size_t at2, T value);
        void fill_randn(T mean, T stddev);

        T operator()(size_t at1, size_t at2);
    protected:
        size_t index(size_t at1, size_t at2);
    private:
        bool transposed = false;
        size_t _size1;
        size_t _size2;
        compute::context _context;
        compute::vector<T> _data;
    };
}

#endif