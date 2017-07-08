#include "TanhActivation.h"

TanhActivation::TanhActivation()
{

}

std::shared_ptr<viennacl::matrix<float>> TanhActivation::compute(std::shared_ptr<viennacl::matrix<float>> x)
{
    return std::make_shared<viennacl::matrix<float>>(viennacl::linalg::element_tanh(*x));
}

std::shared_ptr<viennacl::matrix<float>> TanhActivation::derivative(std::shared_ptr<viennacl::matrix<float>> x)
{
    auto result = this->compute(x);
    viennacl::matrix<float> ones = viennacl::scalar_matrix<float>((*result).size1(), (*result).size2(), 1);

    return std::make_shared<viennacl::matrix<float>>(ones - viennacl::linalg::element_prod(*result, *result));
}