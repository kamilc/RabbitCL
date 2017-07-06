/*
 * scratch.h
 *
 *  Created on: 6 lip 2017
 *      Author: kamil
 */

#ifndef SCRATCH_H_
#define SCRATCH_H_

#include "viennacl/matrix.hpp"

class Activation {
public:
    Activation();
    viennacl::matrix<float> compute(viennacl::matrix<float> x);
    viennacl::matrix<float> derivative(viennacl::matrix<float> x);
};

class ActivationTanh : public Activation {
public:
    ActivationTanh();
    viennacl::matrix<float> compute(viennacl::matrix<float> x);

    viennacl::matrix<float> derivative(viennacl::matrix<float> x) {
        viennacl::matrix<float> result = this->compute(x);
        viennacl::matrix<float> ones =
        viennacl::scalar_matrix<float>(result.size1(), result.size2(), 1);

        return ones - viennacl::linalg::element_prod(result, result);
    }
};

class Layer {
private:
	int _size;
	Activation _activation;
	Layer _parent;
public:
    Layer(int size, Activation activation) {
    		this->_size = size;
    		this->_activation = activation;
    		this->_parent = NullLayer;
    }
    Layer(Layer parent, int size, Activation activation) {
    		this->_size = size;
    		this->_activation = activation;
    		this->_parent = parent;
    }
};

class NullLayer {
};

class Input : public Layer {
public:
    Input(std::vector<float> input, Activation activation)
    : Layer(input.size(), activation) {

    }
};


#endif /* SCRATCH_H_ */
