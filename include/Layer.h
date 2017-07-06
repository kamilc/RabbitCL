#ifndef Layer_h
#define Layer_h

#include <stdio.h>
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/prod.hpp"

class Activation {
public:
    Activation();
    viennacl::matrix<float> compute(viennacl::matrix<float> x);
    viennacl::matrix<float> derivative(viennacl::matrix<float> x);
};

class IdentityActivation : public Activation {
public:
    IdentityActivation();
    viennacl::matrix<float> compute(viennacl::matrix<float> x) {
        return x;
    }
    viennacl::matrix<float> derivative(viennacl::matrix<float> x) {
        return x;
    }
};

class ActivationTanh : public Activation {
public:
    ActivationTanh();
    viennacl::matrix<float> compute(viennacl::matrix<float> x);
    viennacl::matrix<float> derivative(viennacl::matrix<float> x);
};

class BaseLayer {

};

class Layer : public BaseLayer {
private:
    int _size;
    BaseLayer _parent;
    Activation _activation;
public:
    Layer(int size, Activation activation);
    Layer(BaseLayer parent, int size, Activation activation);
};

#endif
