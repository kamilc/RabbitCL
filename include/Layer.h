#ifndef Layer_h
#define Layer_h

// We're using the OpenCL backend:
#define VIENNACL_WITH_OPENCL 1

#include <stdio.h>
#include <boost/optional.hpp>
#include "Activation.h"
#include "TrainDef.h"

class Layer {
private:
    std::size_t _size;
    boost::optional<Layer&> _parent;
    Activation& _activation;
    boost::optional<viennacl::matrix<float>> _synapses;
public:
    Layer(std::size_t size, Activation& activation);
    Layer(Layer& parent, std::size_t size, Activation& activation);

    std::size_t size();
    std::size_t totalSize();
    Layer& root();

    void train(TrainDef& train);

    viennacl::matrix<float> parentForward(viennacl::matrix<float> input);
    viennacl::matrix<float> forward(viennacl::matrix<float> input);
};

#endif
