#include "viennacl/matrix.hpp"
#include "viennacl/linalg/prod.hpp"

#include "Layer.h"

Layer::Layer(int size, Activation& activation) : _activation(activation)
{
    this->_size = size;
}

Layer::Layer(Layer& parent, int size, Activation& activation) : _activation(activation)
{
    this->_parent = parent;
    this->_size = size;
}

int Layer::size()
{
    return this->_size;
}

int Layer::totalSize()
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