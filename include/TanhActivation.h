#ifndef TanhActivation_h
#define TanhActivation_h

#include "viennacl/matrix.hpp"
#include "viennacl/linalg/prod.hpp"
#include "Activation.h"

class TanhActivation : public Activation {
public:
    TanhActivation();
    std::shared_ptr<viennacl::matrix<float>> compute(std::shared_ptr<viennacl::matrix<float>> x);
    std::shared_ptr<viennacl::matrix<float>> derivative(std::shared_ptr<viennacl::matrix<float>> x);
};

#endif