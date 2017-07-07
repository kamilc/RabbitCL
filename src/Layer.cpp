#include "viennacl/matrix.hpp"
#include "viennacl/linalg/prod.hpp"

#include "Layer.h"
#include "Distributions.h"
#include "Memory.h"

Layer::Layer(std::size_t size, Activation& activation) : _activation(activation)
{
    this->_size = size;
}

Layer::Layer(Layer& parent, std::size_t size, Activation& activation) : _activation(activation)
{
    auto synapses = Distributions::scaled_uniform_matrix(parent.size(), size);

    this->_parent = parent;
    this->_size = size;
    this->_synapses = Memory::convert(synapses);
}

std::size_t Layer::size()
{
    return this->_size;
}

std::size_t Layer::totalSize()
{
    if(this->_parent)
    {
        return this->_parent.get().totalSize() * this->_size;
    }
    else
    {
        return this->_size;
    }
}

viennacl::matrix<float> Layer::parentForward(viennacl::matrix<float> input)
{
    if(this->_parent)
    {
        return this->_parent->forward(input);
    }
    else
    {
        return input;
    }
}

viennacl::matrix<float> Layer::forward(viennacl::matrix<float> gpuInput)
{
    viennacl::matrix<float> parentInput = this->parentForward(gpuInput);
    viennacl::matrix<float> weighted = viennacl::linalg::prod(parentInput, this->_synapses.get());

    return this->_activation.compute(weighted);
}

Layer& Layer::root()
{
    if(this->_parent)
    {
        return this->_parent->root();
    }
    else
    {
        return *this;
    }
}

void Layer::train(TrainDef& train)
{
    for(int iter = 1; iter <= train.getEpochs(); iter++) {
        std::cout << "Epoch! " << iter << std::endl;

        viennacl::matrix<float> gpuInput = Memory::convert(train.getInput());
        viennacl::matrix<float> output = this->forward(gpuInput);
        //viennacl::matrix<float> errors = this->error()

    }
}