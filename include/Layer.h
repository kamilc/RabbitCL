#ifndef Layer_h
#define Layer_h

// We're using the OpenCL backend:
#define VIENNACL_WITH_OPENCL 1

#include <stdio.h>
#include <boost/optional.hpp>
#include "matrix.h"

namespace heed
{
    template<typename T, mode MODE>
    class layer
    {
    protected:
        std::size_t _size;
        boost::optional<layer<T, MODE>&> _input;

        // if we don't have any inputs, that means we are at the input
        // level so we don't have any weights. they should be optional
        // then:
        boost::optional<matrix<T, MODE>> _weights;
    public:
        layer(std::size_t size);
        layer(std::size_t size, layer &input);
        layer(layer &input);

        std::size_t size();

        virtual matrix<T, MODE> forward(matrix<T, MODE> &data) = 0;
        void initialize_weights();
    };
}

#endif
