#include "Memory.h"

using GpuMatrix = viennacl::matrix<float>;

std::shared_ptr<GpuMatrix> Memory::convert(boost::numeric::ublas::matrix<float> input)
{
    std::shared_ptr<viennacl::matrix<float>> gpuInput =
        std::make_shared<viennacl::matrix<float>>(viennacl::zero_matrix<float>(input.size1(), input.size2()));
    viennacl::copy(input, *gpuInput);

    return gpuInput;
}