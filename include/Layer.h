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
    private:
        std::size_t _size;
        boost::optional<std::shared_ptr<layer<T, MODE>>> _input;

        // if we don't have any inputs, that means we are at the input
        // level so we don't have any weights. they should be optional
        // then:
        boost::optional<matrix<T, MODE>> _weights;
    public:
        layer(std::size_t size);
        layer(std::size_t size, std::shared_ptr<layer> input);
        layer(std::shared_ptr<layer> input);

        std::size_t size();

        virtual std::shared_ptr<matrix<T, MODE>> forward(std::shared_ptr<matrix<T, MODE>> data) = 0;
        void initializeWeights();
    };
}

#endif
