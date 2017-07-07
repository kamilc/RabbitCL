#ifndef Layer_h
#define Layer_h

// We're using the OpenCL backend:
#define VIENNACL_WITH_OPENCL 1

#include <stdio.h>
#include <valarray>
#include <boost/optional.hpp>

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
public:
    virtual int size();
    virtual int totalSize();

    virtual void train(std::valarray<float> input);
};

class Layer : public BaseLayer {
private:
    int _size;
    boost::optional<BaseLayer&> _parent;
    Activation _activation;
public:
    Layer(int size, Activation activation);
    Layer(BaseLayer& parent, int size, Activation activation);

    int size();
    int totalSize();
};

#endif
