#include "function/reduce_avg.h"

namespace mozart {
    namespace function {

        template<typename T>
        inline const char * reduce_avg_kernel<T>::name()
        {
            return "reduce_avg_kernel";
        }

        template<typename T>
        inline const char * reduce_avg_kernel<T>::code()
        {
            return  "__kernel void reduce_avg_kernel(\n"
                    "          __global float * in,\n"
                    "          __global float * out,\n"
                    "              unsigned int size1,\n"
                    "              unsigned int size2,\n"
                    "              unsigned int isize1)\n"
                    "{ \n"
                    "  out[0] = 0.43f;\n"
                    "};\n";
        }

        template<typename T>
        void reduce_avg_kernel<T>::compute_scalar(matrix<T>& in, scalar<T>& out)
        {
            ocl::enqueue(this->_kernel(in,
                                       out,
                                       cl_uint(in.size1()),
                                       cl_uint(in.size2()),
                                       cl_uint(in.internal_size1())));
            finish();
        }

        template<typename T>
        scalar<T> reduce_avg(matrix<T>& in)
        {
            scalar<T> out(0);

            reduce_avg_kernel<T>::run_scalar(in, out, in.size1(), in.size2() * in.size1());

            return out;
        }

        INSTANTIATE(reduce_avg_kernel);

        template scalar<float> reduce_avg(matrix<float>& in);
        template scalar<double> reduce_avg(matrix<double>& in);
    }
}