#include "function/element_add_assign.h"

namespace mozart
{
    KERNEL(element_add_assign_kernel,
        __kernel void element_add_assign_kernel(              
                    __global TYPE * lhs,
                    __global TYPE * rhs,
                  const struct matrix_size lhs_size,
                  const struct matrix_size rhs_size)
        {
            unsigned int global_id  = get_global_id(0);
            unsigned int total_size = lhs_size.size1 * lhs_size.size2;

            if(global_id < total_size)                
            {
                unsigned int lhs_id = id_to_internal_id(global_id, &lhs_size);
                unsigned int rhs_id = id_to_internal_id(global_id, &rhs_size);

                lhs[lhs_id] += rhs[rhs_id];
            }                                         
        }
    )

    namespace function
    {
        template<typename T>
        void element_add_assign(const matrix<T>& lhs, const matrix<T>& rhs)
        {
            assert(lhs.size1() == rhs.size1() && lhs.size2() == rhs.size2());

            kernel<T, element_add_assign_kernel>::instance()
                .with_global_size(lhs.total_size())
                .with_local_size(lhs.size2())
                .run(
                    lhs,
                    rhs,
                    lhs.size(),
                    rhs.size()
                );
        }

        template void element_add_assign(const matrix<float>& lhs, const matrix<float>& rhs);
        template void element_add_assign(const matrix<double>& lhs, const matrix<double>& rhs);
    }
}