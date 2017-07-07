#ifndef TanhActivation_h
#define TanhActivation_h

#include "viennacl/matrix.hpp"
#include "viennacl/linalg/prod.hpp"
#include "Activation.h"

class TanhActivation : public Activation {
public:
    TanhActivation();
    viennacl::matrix<float> compute(viennacl::matrix<float> x);
    viennacl::matrix<float> derivative(viennacl::matrix<float> x);
};

#endif