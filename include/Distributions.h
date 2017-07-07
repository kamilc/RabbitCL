#ifndef Distributions_h
#define Distributions_h

#include <boost/numeric/ublas/matrix.hpp>

class Distributions {
public:
    static boost::numeric::ublas::matrix<float> scaled_uniform_matrix(std::size_t size1, std::size_t size2);
};

#endif