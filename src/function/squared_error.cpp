#include "function/squared_error.h"

namespace mozart
{
    namespace function
    {
        template<typename T>
        inline const char * squared_error_kernel<T>::name()
        {
            return "squared_error_kernel";
        }

        template<typename T>
        inline const char * squared_error_kernel<T>::code()
        {
            return  "__kernel void squared_error_kernel(\n"
                    "          __global float * yhat, \n"
                    "          __global float * targets, \n"
                    "          __global float * out,\n"
                    "              unsigned int size1,\n"
                    "              unsigned int size2,\n"
                    "              unsigned int isize1)\n"
                    "{ \n"
                    "  unsigned int gid = get_global_id(0);\n"
                    "  unsigned int padded = isize1 - size1;\n"
                    "  unsigned int row = gid / size2;\n"
                    "  unsigned int idx = gid + row*padded;\n"
                    "  float diff = targets[idx] - yhat[idx];"
                    "  out[idx] = 0.5*diff*diff;\n"
                    "};\n";
        }

        template<typename T>
        void squared_error_kernel<T>::compute_matrix(matrix<T>& in, matrix_range<matrix<T>>& targets, matrix<T>& out)
        {
            ocl::enqueue(this->_kernel(in,
                                       targets,
                                       out,
                                       cl_uint(out.size1()),
                                       cl_uint(out.size2()),
                                       cl_uint(out.internal_size1())));
            finish();
        }

        template<typename T>
        inline const char * squared_error_deriv_kernel<T>::name()
        {
            return "squared_error_deriv_kernel";
        }

        template<typename T>
        inline const char * squared_error_deriv_kernel<T>::code()
        {
            // todo: add tests!
            return  "__kernel void squared_error_deriv_kernel(\n"
                    "          __global float * yhat, \n"
                    "          __global float * targets, \n"
                    "          __global float * out,\n"
                    "              unsigned int size1,\n"
                    "              unsigned int size2,\n"
                    "              unsigned int isize1)\n"
                    "{ \n"
                    "  unsigned int gid = get_global_id(0);\n"
                    "  unsigned int padded = isize1 - size1;\n"
                    "  unsigned int row = gid / size2;\n"
                    "  unsigned int idx = gid + row*padded;\n"
                    "  out[idx] = targets[idx] - yhat[idx];"
                    "};\n";
        }

        template<typename T>
        void squared_error_deriv_kernel<T>::compute_matrix(matrix<T>& in, matrix_range<matrix<T>>& targets, matrix<T>& out)
        {
            ocl::enqueue(this->_kernel(in,
                                       targets,
                                       out,
                                       cl_uint(out.size1()),
                                       cl_uint(out.size2()),
                                       cl_uint(out.internal_size1())));
            finish();
        }

        template<typename T>
        cost<T> squared_error(matrix<T>& in, matrix_range<matrix<T>>& targets, bool derive)
        {
            cost<T> result(in, derive);

            squared_error_kernel<T>::run_matrix(in, targets, result.out, in.size1(), in.size2() * in.size1());

            if(derive)
            {
                squared_error_deriv_kernel<T>::run_matrix(in, targets, result.deriv, in.size1(), in.size2() * in.size1());
            }

            return result;
        }

        INSTANTIATE(squared_error_kernel);
        INSTANTIATE(squared_error_deriv_kernel);

        template cost<float> squared_error(matrix<float>& in, matrix_range<matrix<float>>& targets, bool derive);
        template cost<double> squared_error(matrix<double>& in, matrix_range<matrix<double>>& targets, bool derive);
    }
}