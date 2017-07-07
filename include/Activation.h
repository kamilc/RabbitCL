#ifndef Activation_h
#define Activation_h

#include "viennacl/matrix.hpp"

class Activation {
public:
    Activation();
    viennacl::matrix<float> compute(viennacl::matrix<float> x);
    viennacl::matrix<float> derivative(viennacl::matrix<float> x);
};

#endif