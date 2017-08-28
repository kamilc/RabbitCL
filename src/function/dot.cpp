#include "function/dot.h"

namespace mozart
{
    namespace function
    {
        template<typename T>
        matrix<T> dot(matrix<T>& lhs, matrix<T>& rhs)
        {
            assert(lhs.size2() == rhs.size1());

            matrix<T> out(lhs.size1(), rhs.size2());

            // improve: implement me in a clean way:
            compute::event _event;
            compute::command_queue queue = context_manager::instance().new_queue();

            auto err = clblasSetup( );

            auto lhs_size = lhs.size();
            auto rhs_size = rhs.size();

            auto M = lhs_size.size1;
            auto N = rhs_size.size2;
            auto K = lhs_size.size2;
            auto alpha = 1;
            auto bufA = lhs.data().get_buffer().get();
            auto lda = lhs_size.internal_size2;
            auto bufB = rhs.data().get_buffer().get();
            auto ldb = rhs_size.internal_size2;
            auto beta = 1;
            auto bufC = out.data().get_buffer().get();
            auto ldc = rhs_size.internal_size2;

            err = clblasSgemm( clblasRowMajor, clblasNoTrans, clblasNoTrans,
                M, N, K,
                alpha, bufA, lhs.offset(), lda,
                bufB, rhs.offset(), ldb, beta,
                bufC, 0, ldc,
                1, &queue.get(), 0, NULL, &_event.get() );

            _event.wait();

            return out;
        }

        template matrix<float> dot(matrix<float>& lhs, matrix<float>& rhs);
        template matrix<double> dot(matrix<double>& lhs, matrix<double>& rhs);
    }
}
