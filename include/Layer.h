#ifndef Layer_h
#define Layer_h

// We're using the OpenCL backend:
#define VIENNACL_WITH_OPENCL 1

#include <stdio.h>
#include <boost/optional.hpp>
#include "matrix.h"

namespace heed
{

    template<typename T>
    class layer
    {
    private:
        std::size_t _size;
        std::vector<std::shared_ptr<layer<T>>> _inputs;
    public:
        layer(std::size_t size);
        layer(std::size_t size, std::shared_ptr<layer> input);
        layer(std::size_t size, std::vector<std::shared_ptr<layer>> inputs);
        layer(std::shared_ptr<layer> input);
        layer(std::vector<std::shared_ptr<layer>> inputs);

        std::size_t size();

        virtual std::shared_ptr<matrix<T>> forward(std::shared_ptr<matrix<T>> data) = 0;
    };
}

#endif
