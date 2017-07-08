#include "Distributions.h"

boost::numeric::ublas::matrix<float> Distributions::scaled_uniform_matrix(std::size_t size1, std::size_t size2)
{
    return boost::numeric::ublas::matrix<float>(size1, size2);
}