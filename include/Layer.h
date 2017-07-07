#ifndef Layer_h
#define Layer_h

// We're using the OpenCL backend:
#define VIENNACL_WITH_OPENCL 1

#include <stdio.h>
#include <valarray>
#include <boost/optional.hpp>
#include "Activation.h"

class Layer {
private:
    int _size;
    boost::optional<Layer&> _parent;
    Activation& _activation;
public:
    Layer(int size, Activation& activation);
    Layer(Layer& parent, int size, Activation& activation);

    int size();
    int totalSize();
};

#endif
