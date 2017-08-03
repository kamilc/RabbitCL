#ifndef Layer_h
#define Layer_h

// We're using the OpenCL backend:
#define VIENNACL_WITH_OPENCL 1

#include <stdio.h>
#include <boost/optional.hpp>

#include "utilities.h"
#include "matrix.h"
#include "layer_config.h"

namespace heed
{
    template<typename T, mode MODE>
    class layer
    {
    protected:
        std::size_t _size;
        boost::optional<std::size_t> _parent_size;
        boost::optional<matrix<T, MODE>> _weights;
    public:
        layer(layer_config<T, MODE> &config);

        std::size_t size();

        virtual matrix<T, MODE> forward(matrix<T, MODE> &data) = 0;
        void initialize_weights();
    };
}

#endif
