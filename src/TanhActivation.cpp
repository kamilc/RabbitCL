#include "TanhActivation.h"

TanhActivation::TanhActivation()
{

}

viennacl::matrix<float> TanhActivation::compute(viennacl::matrix<float> x)
{
    return viennacl::linalg::element_tanh(x);
}

viennacl::matrix<float> TanhActivation::derivative(viennacl::matrix<float> x)
{
    viennacl::matrix<float> result = this->compute(x);
    viennacl::matrix<float> ones =
    viennacl::scalar_matrix<float>(result.size1(), result.size2(), 1);

    return ones - viennacl::linalg::element_prod(result, result);
}