// We're using the OpenCL backend:
#define VIENNACL_WITH_OPENCL 1

#include "viennacl/matrix.hpp"
#include "viennacl/linalg/prod.hpp"

class Activation {
public:
    Activation() {

    }
    viennacl::matrix<float> compute(viennacl::matrix<float> x) {
        return x;
    }
    viennacl::matrix<float> derivative(viennacl::matrix<float> x) {
        return x;
    }
};

class ActivationTanh : public Activation {
public:
    ActivationTanh() : Activation() {

    }
    viennacl::matrix<float> compute(viennacl::matrix<float> x) {
        return viennacl::linalg::element_tanh(x);
    }

    viennacl::matrix<float> derivative(viennacl::matrix<float> x) {
        viennacl::matrix<float> result = this->compute(x);
        viennacl::matrix<float> ones =
        viennacl::scalar_matrix<float>(result.size1(), result.size2(), 1);

        return ones - viennacl::linalg::element_prod(result, result);
    }
};

class Layer {
public:
    Layer(int size, Activation activation) {

    }
    Layer(Layer parent, int size, Activation activation) {

    }
};

class Input : public Layer {
public:
    Input(std::vector<float> input, Activation activation)
    : Layer(input.size(), activation) {

    }
};

