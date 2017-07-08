#ifndef IdentityActivation_h
#define IdentityActivation_h

#include "viennacl/matrix.hpp"
#include "Activation.h"

class IdentityActivation : public Activation {
public:
    IdentityActivation();
    std::shared_ptr<viennacl::matrix<float>> compute(std::shared_ptr<viennacl::matrix<float>> x);
    std::shared_ptr<viennacl::matrix<float>> derivative(std::shared_ptr<viennacl::matrix<float>> x);
};

#endif