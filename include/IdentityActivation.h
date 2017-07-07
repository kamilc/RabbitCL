#ifndef IdentityActivation_h
#define IdentityActivation_h

#include "viennacl/matrix.hpp"
#include "Activation.h"

class IdentityActivation : public Activation {
public:
    IdentityActivation();
    viennacl::matrix<float> compute(viennacl::matrix<float> x);
    viennacl::matrix<float> derivative(viennacl::matrix<float> x);
};

#endif