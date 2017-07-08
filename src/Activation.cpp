#include "Activation.h"

Activation::Activation()
{

}

std::shared_ptr<viennacl::matrix<float>> Activation::compute(std::shared_ptr<viennacl::matrix<float>> x)
{
    return x;
}

std::shared_ptr<viennacl::matrix<float>> Activation::derivative(std::shared_ptr<viennacl::matrix<float>> x)
{
    return x;
}