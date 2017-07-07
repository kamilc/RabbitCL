#include "IdentityActivation.h"

IdentityActivation::IdentityActivation()
{

}

viennacl::matrix<float> IdentityActivation::compute(viennacl::matrix<float> x)
{
    return x;
}

viennacl::matrix<float> IdentityActivation::derivative(viennacl::matrix<float> x)
{
    return x;
}