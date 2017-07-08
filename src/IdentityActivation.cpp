#include "IdentityActivation.h"

IdentityActivation::IdentityActivation()
{

}

std::shared_ptr<viennacl::matrix<float>> IdentityActivation::compute(std::shared_ptr<viennacl::matrix<float>> x)
{
    return x;
}

std::shared_ptr<viennacl::matrix<float>> IdentityActivation::derivative(std::shared_ptr<viennacl::matrix<float>> x)
{
    return x;
}