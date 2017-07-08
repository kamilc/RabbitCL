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
    auto data = Memory::convert(synapses);

    this->_parent = parent;
    this->_size = size;
    this->_synapses = data;

    std::cout << "Initialized!" << std::endl;
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

std::shared_ptr<viennacl::matrix<float>> Layer::parentForward(std::shared_ptr<viennacl::matrix<float>> input)
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

std::shared_ptr<viennacl::matrix<float>> Layer::forward(std::shared_ptr<viennacl::matrix<float>> gpuInput)
{
    if(this->_parent)
    {
        std::shared_ptr<viennacl::matrix<float>> parentInput = this->parentForward(gpuInput);
        viennacl::matrix<float> weighted = viennacl::linalg::prod(*parentInput, *(this->_synapses.get()));

        return this->_activation.compute(std::make_shared<viennacl::matrix<float>>(weighted));
    }
    else 
    {
        return gpuInput;
    }
    
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

        std::shared_ptr<viennacl::matrix<float>> gpuInput = Memory::convert(train.getInput());
        std::shared_ptr<viennacl::matrix<float>> output = this->forward(gpuInput);
        //viennacl::matrix<float> errors = this->error()

    }
}