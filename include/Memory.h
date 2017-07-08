#ifndef Memory_h
#define Memory_h

#include <boost/numeric/ublas/matrix.hpp>
#include "viennacl/matrix.hpp"

class Memory {
public:
    static std::shared_ptr<viennacl::matrix<float>> convert(boost::numeric::ublas::matrix<float> input);
};

#endif