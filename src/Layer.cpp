#include "viennacl/matrix.hpp"
#include "viennacl/linalg/prod.hpp"

#include "Layer.h"
#include "IdentityLayer.h"

Activation::Activation()
{

}

viennacl::matrix<float> Activation::compute(viennacl::matrix<float> x)
{
    return x;
}

viennacl::matrix<float> Activation::derivative(viennacl::matrix<float> x)
{
    return x;
}


ActivationTanh::ActivationTanh()
{

}

viennacl::matrix<float> ActivationTanh::compute(viennacl::matrix<float> x)
{
    return viennacl::linalg::element_tanh(x);
}

viennacl::matrix<float> ActivationTanh::derivative(viennacl::matrix<float> x)
{
    viennacl::matrix<float> result = this->compute(x);
    viennacl::matrix<float> ones =
    viennacl::scalar_matrix<float>(result.size1(), result.size2(), 1);

    return ones - viennacl::linalg::element_prod(result, result);
}

Layer::Layer(int size, Activation activation)
{
    this->_size = size;
    this->_activation = activation;
}

Layer::Layer(BaseLayer& parent, int size, Activation activation)
{
    this->_parent = parent;
    this->_size = size;
    this->_activation;
}

int BaseLayer::size()
{
    return 1;
}

int BaseLayer::totalSize()
{
    std::cout << "BaseLayer::totalSize!" << std::endl;

    return 1;
}

void BaseLayer::train(std::valarray<float> input) {
    return;
}

int Layer::size()
{
    return this->_size;
}

int Layer::totalSize()
{
    std::cout << "Layer::totalSize!" << std::endl;

    if(this->_parent)
    {
        return this->_parent.get().totalSize() * this->_size;
    }
    else
    {
        return this->_size;
    }
}