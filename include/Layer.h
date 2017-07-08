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
    boost::optional<std::shared_ptr<viennacl::matrix<float>>> _synapses;
public:
    Layer(std::size_t size, Activation& activation);
    Layer(Layer& parent, std::size_t size, Activation& activation);

    std::size_t size();
    std::size_t totalSize();
    Layer& root();

    void train(TrainDef& train);

    std::shared_ptr<viennacl::matrix<float>> parentForward(std::shared_ptr<viennacl::matrix<float>> input);
    std::shared_ptr<viennacl::matrix<float>> forward( std::shared_ptr<viennacl::matrix<float>> input);
};

#endif
