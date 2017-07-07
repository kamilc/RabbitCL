#include "Memory.h"

viennacl::matrix<float> Memory::convert(boost::numeric::ublas::matrix<float> input)
{
    viennacl::matrix<float> gpuInput =
        viennacl::zero_matrix<float>(input.size1(), input.size2());
    viennacl::copy(input, gpuInput);

    return gpuInput;
}