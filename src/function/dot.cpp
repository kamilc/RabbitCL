#include "function/dot.h"

namespace mozart
{
    namespace function
    {
        template<typename T>
        matrix<T> dot(matrix<T>& lhs, matrix<T>& rhs, bool lhs_transpose, bool rhs_transpose)
        {
            auto lhs_size2 = lhs_transpose ? lhs.size1() : lhs.size2();
            auto rhs_size1 = rhs_transpose ? rhs.size2() : rhs.size1();

            assert(lhs_size2 == rhs_size1);

            auto lhs_size1 = lhs_transpose ? lhs.size2() : lhs.size1();
            auto rhs_size2 = rhs_transpose ? rhs.size1() : rhs.size2();

            matrix<T> out(lhs_size1, rhs_size2);

            compute::event _event;
            compute::command_queue queue = context_manager::instance().new_queue();

            auto lhs_size = lhs.size();
            auto rhs_size = rhs.size();

            auto M = lhs_transpose ? lhs_size.size2 : lhs_size.size1;
            auto N = rhs_transpose ? rhs_size.size1 : rhs_size.size2;
            auto K = lhs_transpose ? lhs_size.size1 : lhs_size.size2;
            auto alpha = 1;
            auto bufA = lhs.data().get_buffer().get();
            auto lda = lhs_size.internal_size2;
            auto bufB = rhs.data().get_buffer().get();
            auto ldb = rhs_size.internal_size2;
            auto beta = 1;
            auto bufC = out.data().get_buffer().get();
            auto ldc = rhs_transpose ? rhs_size.internal_size1 : rhs_size.internal_size2;
            auto ltrans = lhs_transpose ? clblast::Transpose::kYes : clblast::Transpose::kNo;
            auto rtrans = rhs_transpose ? clblast::Transpose::kYes : clblast::Transpose::kNo;

            auto err = clblast::Gemm<T>(clblast::Layout::kRowMajor, ltrans, rtrans,
                M, N, K,
                alpha,
                bufA, lhs.offset(), lda,
                bufB, rhs.offset(), ldb,
                beta,
                bufC, 0, ldc,
                &queue.get(), &_event.get());

            if((int)err != 0)
            {
                std::cout << "dot result code: " << (int)err << std::endl;
            }

            _event.wait();
            queue.finish();

            return out;
        }

        template matrix<float> dot(matrix<float>& lhs, matrix<float>& rhs, bool lhs_transpose, bool rhs_transpose);
        template matrix<double> dot(matrix<double>& lhs, matrix<double>& rhs, bool lhs_transpose, bool rhs_transpose);
    }
}
