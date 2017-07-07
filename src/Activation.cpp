#include "Activation.h"

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