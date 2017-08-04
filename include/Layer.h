#ifndef Layer_h
#define Layer_h

#include <stdio.h>
#include <boost/optional.hpp>
#include "viennacl/matrix.hpp"

#include "utilities.h"
#include "layer_config.h"

using namespace viennacl;
using namespace std;
using namespace boost;

namespace mozart
{
    template<typename T>
    class layer
    {
    protected:
        size_t _size;
        optional<size_t> _parent_size;
        optional<matrix<T>> _weights;
    public:
        layer(layer_config<T> &config);

        size_t size();

        virtual matrix<T> forward(matrix<T> &data) = 0;
        void initialize_weights();
    };
}

#endif
