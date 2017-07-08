#ifndef Activation_h
#define Activation_h

#include "viennacl/matrix.hpp"

class Activation {
public:
    Activation();
    std::shared_ptr<viennacl::matrix<float>> compute(std::shared_ptr<viennacl::matrix<float>> x);
    std::shared_ptr<viennacl::matrix<float>> derivative(std::shared_ptr<viennacl::matrix<float>> x);
};

#endif