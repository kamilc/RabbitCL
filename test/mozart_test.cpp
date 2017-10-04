#include <iostream>

#define BOOST_COMPUTE_DEBUG_KERNEL_COMPILATION 1

#include <boost/compute.hpp>

#include <chrono>
#include <cmath>
#include "gtest/gtest.h"
#include "sequence.h"
#include "input_config.h"
#include "dense_config.h"
#include "function/tanh.h"
#include "function/relu.h"
#include "function/softmax.h"
#include "function/squared_error.h"
#include "function/categorical_cross_entropy.h"
#include "matrix_helpers.h"
#include "function/reduce_avg.h"
#include "function/dot.h"
#include "function/element_mul.h"
#include "function/squashmax.h"
#include "function/adagrad_update.h"

using namespace mozart;
using namespace mozart::function;

TEST(element_mul_test, element_mul_test)
{
    auto data1 = make_matrix<float>({
        { 7, 2, 8.0, 4 },
        { 1, 2, 3.0, 2 },
        { 6, 7, 9.5, 3 },
        { 1, 2, 3.0, 4 },
        { 4, 5, 6.0, 5 }
    });

    auto data2 = make_matrix<float>({
        { 1, 2, 3.0, 2 },
        { 1, 2, 3.0, 4 },
        { 7, 2, 8.0, 4 },
        { 4, 5, 6.0, 5 },
        { 6, 7, 9.5, 3 }
    });

    auto view1 = matrix<float>::view(data1, 1, 3, 1, 3);
    auto view2 = matrix<float>::view(data2, 1, 3, 1, 3);

    auto result = element_mul(view1, view2);

    EXPECT_NEAR(result(0, 0), 4, 0.0001);
    EXPECT_NEAR(result(0, 1), 9, 0.0001);
    EXPECT_NEAR(result(0, 2), 8, 0.0001);
    EXPECT_NEAR(result(1, 0), 14, 0.0001);
    EXPECT_NEAR(result(1, 1), 76, 0.0001);
    EXPECT_NEAR(result(1, 2), 12, 0.0001);
    EXPECT_NEAR(result(2, 0), 10, 0.0001);
    EXPECT_NEAR(result(2, 1), 18, 0.0001);
    EXPECT_NEAR(result(2, 2), 20, 0.0001);
}

TEST(scale_test_case, scale_view_test)
{
    auto data1 = make_matrix<float>({
        { 7, 2, 8.0, 4 },
        { 1, 2, 3.0, 2 },
        { 6, 7, 9.5, 3 },
        { 1, 2, 3.0, 4 },
        { 4, 5, 6.0, 5 }
    });

    auto data_view = matrix<float>::view(data1, 1, 2, 0, 2);

    auto factor = scalar<float>(0.5);

    matrix<float> result = scale<float>(data_view, factor);

    EXPECT_NEAR(result(0, 0), 0.5, 0.0001);
    EXPECT_NEAR(result(0, 1), 1.0, 0.0001);
    EXPECT_NEAR(result(0, 2), 1.5, 0.0001);
    EXPECT_NEAR(result(1, 0), 3.0, 0.0001);
    EXPECT_NEAR(result(1, 1), 3.5, 0.0001);
    EXPECT_NEAR(result(1, 2), 4.75, 0.0001);
}


TEST(dot_test_case, dot_test)
{
    auto data1 = make_matrix<float>({
        { 1, 2, 3.0 },
        { 6, 7, 9.5 }
    });

    auto data2 = make_matrix<float>({
        { 1, },
        { 6, },
        { 5 }
    });

    matrix<float> result = dot<float>(data1, data2);

    EXPECT_NEAR(result(0, 0), 28.0, 0.0001);
    EXPECT_NEAR(result(1, 0), 95.5, 0.0001);
}

TEST(dot_test_case, dot_view_test)
{
    auto data1 = make_matrix<float>({
        { 7, 2, 8.0, 4 },
        { 1, 2, 3.0, 2 },
        { 6, 7, 9.5, 3 },
        { 1, 2, 3.0, 4 },
        { 4, 5, 6.0, 5 }
    });

    auto data_view = matrix<float>::view(data1, 1, 2, 0, 2);
    auto data_view2 = matrix<float>::view(data1, 2, 4, 1, 3);

    auto data2 = make_matrix<float>({
        { 1, },
        { 6, },
        { 5 }
    });

    matrix<float> result = dot<float>(data_view, data2);
    matrix<float> result2 = dot<float>(data_view2, data2);

    EXPECT_NEAR(result(0, 0), 28.0, 0.0001);
    EXPECT_NEAR(result(1, 0), 95.5, 0.0001);

    EXPECT_NEAR(result2(0, 0), 79.0, 0.0001);
    EXPECT_NEAR(result2(1, 0), 40.0, 0.0001);
    EXPECT_NEAR(result2(2, 0), 66.0, 0.0001);
}

TEST(dot_test_case, dot_transpose_test)
{
    auto lhs = make_matrix<float>({ // 3x2
        {1, 2},
        {3, 4},
        {5, 6}
    });

    auto rhs = make_matrix<float>({ // 3x2
        {1, 2},
        {3, 4},
        {5, 6}
    });

    // we really need 2x3 X 3x2

    matrix<float> result = dot<float>(lhs, rhs, true, false);

    EXPECT_NEAR(result(0, 0), 35.0, 0.0001);
    EXPECT_NEAR(result(0, 1), 44.0, 0.0001);
    EXPECT_NEAR(result(1, 0), 44.0, 0.0001);
    EXPECT_NEAR(result(1, 1), 56.0, 0.0001);
}

TEST(dot_test_case, dot_transpose2_test)
{
    auto lhs = make_matrix<float>({ // 3x2
        {1, 2},
        {3, 4},
        {5, 6}
    });

    auto rhs = make_matrix<float>({ // 3x2
        {1, 2},
        {3, 4},
        {5, 6}
    });

    // we really need 3x2 X 2x3

    matrix<float> result = dot<float>(lhs, rhs, false, true);

    EXPECT_NEAR(result(0, 0), 5.0, 0.0001);
    EXPECT_NEAR(result(0, 1), 11.0, 0.0001);
    EXPECT_NEAR(result(0, 2), 17.0, 0.0001);
    EXPECT_NEAR(result(1, 0), 11.0, 0.0001);
    EXPECT_NEAR(result(1, 1), 25.0, 0.0001);
    EXPECT_NEAR(result(1, 2), 39.0, 0.0001);
    EXPECT_NEAR(result(2, 0), 17.0, 0.0001);
    EXPECT_NEAR(result(2, 1), 39.0, 0.0001);
    EXPECT_NEAR(result(2, 2), 61.0, 0.0001);
}

TEST(dot_test_case, dot_transpose3_test)
{
    auto lhs = make_matrix<float>({ // 2x3
        {1, 2, 3},
        {4, 5, 6}
    });

    auto rhs = make_matrix<float>({ // 3x2
        {1, 2},
        {3, 4},
        {5, 6}
    });

    // we really need 3x2 X 2x3

    matrix<float> result = dot<float>(lhs, rhs, true, true);

    EXPECT_NEAR(result(0, 0), 9.0, 0.0001);
    EXPECT_NEAR(result(0, 1), 19.0, 0.0001);
    EXPECT_NEAR(result(0, 2), 29.0, 0.0001);
    EXPECT_NEAR(result(1, 0), 12.0, 0.0001);
    EXPECT_NEAR(result(1, 1), 26.0, 0.0001);
    EXPECT_NEAR(result(1, 2), 40.0, 0.0001);
    EXPECT_NEAR(result(2, 0), 15.0, 0.0001);
    EXPECT_NEAR(result(2, 1), 33.0, 0.0001);
    EXPECT_NEAR(result(2, 2), 51.0, 0.0001);
}

TEST(reduce_avg_test_case, reduce_avg_test)
{
    auto data = make_matrix<float>({
        { 1, 2, 3.0 },
        { 6, 7, 9.5 }
    });

    auto data2 = make_matrix<float>({
        { 1, 2, 3, 4 },
        { 5, 6, 7, 8 },
        { 9, 10, 11, 12 },
        { 13, 14, 15, 16 },
        { 17, 18, 19, 20 },
    });

    auto data3 = make_matrix<float>({
        { 0, 1, 2, 3, 4 },
        { 0, 5, 6, 7, 8 },
        { 0, 9, 10, 11, 12 },
        { 0, 13, 14, 15, 16 },
    });

    auto view = matrix<float>::view(data, 1, 1, 1, 2);

    float result = reduce_avg<float>(data);
    float result2 = reduce_avg<float>(data2);
    float result3 = reduce_avg<float>(data3);
    float result_from_view = reduce_avg<float>(view);

    EXPECT_NEAR(result, 4.75, 0.0001);
    EXPECT_NEAR(result2, 10.5, 0.0001);
    EXPECT_NEAR(result3, 6.8, 0.0001);
    EXPECT_NEAR(result_from_view, 8.25, 0.0001);
}

// TEST(squared_error_test_case, squared_error_test)
// {
//     auto predicted = make_matrix<float>({
//         { 0,  1, -1 },
//         { 2,  1, -10},
//     });

//     auto targets = make_matrix<float>({
//         { 1.5,   -2,   0 },
//         { 2.1,  1.1, -10 }
//     });

//     auto result = squared_error<float>(predicted, targets, false);

//     EXPECT_NEAR(result.out(0, 0), 6.125, 0.0001);
//     EXPECT_NEAR(result.out(1, 0), 0.01, 0.0001);
// }

// TEST(squared_error_test_case, squared_error_deriv_test)
// {
//     auto predicted = make_matrix<float>({
//         { 0,  1, -1 },
//         { 2,  1, -10},
//     });

//     auto targets = make_matrix<float>({
//         { 1.5,   -2,   0 },
//         { 2.1,  1.1, -10 }
//     });

//     auto result = squared_error<float>(predicted, targets, true);

//     EXPECT_NEAR(result.deriv(0, 0), 0.5, 0.0001);
//     EXPECT_NEAR(result.deriv(1, 0), -0.2, 0.0001);
// }

TEST(relu_test_case, relu_test)
{
    auto data = make_matrix<float>({
        {-5, -4, -2 },
        { 0,  1, -1 },
        { 2,  1, -10},
    });

    auto result = relu<float>(data, false);

    EXPECT_EQ(result.out(0, 0), 0);
    EXPECT_EQ(result.out(0, 1), 0);
    EXPECT_EQ(result.out(0, 2), 0);
    EXPECT_EQ(result.out(1, 0), 0);
    EXPECT_EQ(result.out(1, 1), 1);
    EXPECT_EQ(result.out(1, 2), 0);
    EXPECT_EQ(result.out(2, 0), 2);
    EXPECT_EQ(result.out(2, 1), 1);
    EXPECT_EQ(result.out(2, 2), 0);
}

TEST(relu_test_case, relu_deriv_test)
{
    auto data = make_matrix<float>({
        {-5, -4, -2 },
        { 0,  1, -1 },
        { 2,  1, -10},
    });

    auto result = relu<float>(data, true);

    EXPECT_EQ(result.deriv(0, 0), 0);
    EXPECT_EQ(result.deriv(0, 1), 0);
    EXPECT_EQ(result.deriv(0, 2), 0);
    EXPECT_EQ(result.deriv(1, 0), 0);
    EXPECT_EQ(result.deriv(1, 1), 1);
    EXPECT_EQ(result.deriv(1, 2), 0);
    EXPECT_EQ(result.deriv(2, 0), 1);
    EXPECT_EQ(result.deriv(2, 1), 1);
    EXPECT_EQ(result.deriv(2, 2), 0);
}

TEST(tanh_test_case, tanh_test)
{
    auto data = make_matrix<float>({
        {-5, -4, -2 },
        { 0,  1, -1 },
        { 2,  1, -10},
    });

    auto result = tanh<float>(data, false);

    EXPECT_NEAR(result.out(0, 0), -0.9999092, 0.0001);
    EXPECT_NEAR(result.out(0, 1), -0.9993293, 0.0001);
    EXPECT_NEAR(result.out(0, 2), -0.96402758, 0.0001);
    EXPECT_NEAR(result.out(1, 0), 0, 0.0001);
    EXPECT_NEAR(result.out(1, 1), 0.76159416, 0.0001);
    EXPECT_NEAR(result.out(1, 2), -0.76159416, 0.0001);
    EXPECT_NEAR(result.out(2, 0), 0.96402758, 0.0001);
    EXPECT_NEAR(result.out(2, 1), 0.76159416, 0.0001);
    EXPECT_NEAR(result.out(2, 2), -1, 0.0001);
}

TEST(tanh_test_case, tanh_deriv_test)
{
    auto data = make_matrix<float>({
        {-5, -4, -2 },
        { 0,  1, -1 },
        { 2,  1, -10},
    });

    auto result = tanh<float>(data, true);

    EXPECT_NEAR(result.deriv(0, 0), 0.00018158, 0.0001);
    EXPECT_NEAR(result.deriv(0, 1), 0.00134095, 0.0001);
    EXPECT_NEAR(result.deriv(0, 2), 0.07065082, 0.0001);
    EXPECT_NEAR(result.deriv(1, 0), 1, 0.0001);
    EXPECT_NEAR(result.deriv(1, 1), 0.41997434, 0.0001);
    EXPECT_NEAR(result.deriv(1, 2), 0.41997434, 0.0001);
    EXPECT_NEAR(result.deriv(2, 0), 7.06508249e-02, 0.0001);
    EXPECT_NEAR(result.deriv(2, 1), 4.19974342e-01, 0.0001);
    EXPECT_NEAR(result.deriv(2, 2), 8.24461455e-09, 0.0001);
}

TEST(softmax_test_case, softmax_test)
{
    auto data = make_matrix<float>({
        {  3.0, 1.0,  0.2 },
        {  3.0, 1.5, -0.2 },
        { -3.0, 1.5, -0.2 },
    });

    auto view = matrix<float>::view(data, 1, 2, 1, 2);

    auto result = softmax<float>(data, false);
    auto result_view = softmax<float>(view, false);

    EXPECT_NEAR(result.out(0, 0), 0.8360188, 0.0001);
    EXPECT_NEAR(result.out(0, 1), 0.11314284, 0.0001);
    EXPECT_NEAR(result.out(0, 2), 0.05083836, 0.0001);
    EXPECT_NEAR(result.out(1, 0), 0.79120662, 0.0001);
    EXPECT_NEAR(result.out(1, 1), 0.17654206, 0.0001);
    EXPECT_NEAR(result.out(1, 2), 0.03225133, 0.0001);
    EXPECT_NEAR(result.out(2, 0), 0.00930563, 0.0001);
    EXPECT_NEAR(result.out(2, 1), 0.8376665, 0.0001);
    EXPECT_NEAR(result.out(2, 2), 0.15302787, 0.0001);

    EXPECT_NEAR(result_view.out(0, 0), 0.8455347, 0.0001);
    EXPECT_NEAR(result_view.out(0, 1), 0.1544653, 0.0001);
    EXPECT_NEAR(result_view.out(1, 0), 0.8455347, 0.0001);
    EXPECT_NEAR(result_view.out(1, 1), 0.1544653, 0.0001);
}

TEST(softmax_test_case, softmax_deriv_test)
{
    auto data = make_matrix<float>({
        {  3.0, 1.0,  0.2 },
        {  3.0, 1.5, -0.2 },
        { -3.0, 1.5, -0.2 },
    });

    auto result = softmax<float>(data, true);

    EXPECT_NEAR(result.deriv(0, 0), 0.13709136, 0.0001);
    EXPECT_NEAR(result.deriv(0, 1), 0.10034154, 0.0001);
    EXPECT_NEAR(result.deriv(0, 2), 0.04825382, 0.0001);
    EXPECT_NEAR(result.deriv(1, 0), 0.16519871, 0.0001);
    EXPECT_NEAR(result.deriv(1, 1), 0.14537496, 0.0001);
    EXPECT_NEAR(result.deriv(1, 2), 0.03121118, 0.0001);
    EXPECT_NEAR(result.deriv(2, 0), 0.00921904, 0.0001);
    EXPECT_NEAR(result.deriv(2, 1), 0.13598134, 0.0001);
    EXPECT_NEAR(result.deriv(2, 2), 0.12961034, 0.0001);
}

TEST(softmax_test_case, softmax_not_giving_nans)
{
  auto data = make_matrix<float>({
     { 63.4369, 61.1316, 58.974, 63.1129, -8.20094, 9.65623, -0.39024, -8.97584 },
     { 54.412, 56.657, 44.56, 51.11, -6.0202, 6.97188, 0.502755, -6.60736 },
     { 82.9402, 74.3101, 90.0276, 88.3705, -11.8119, 15.4398, 0.127421, -13.086 },
     { 74.3448, 68.5043, 76.3465, 77.2448, -10.2242, 12.8946, -0.100049, -11.2721 }
  });

  auto result = softmax<float>(data, true);

  for(auto row = 0; row < result.out.size1(); row++)
  {
      for(auto col = 0; col < result.out.size2(); col++)
      {
          EXPECT_TRUE(!std::isnan(result.out(row, col)));
      }
  }
}

TEST(matrix_view_test, matrix_view_test)
{
    auto data = make_matrix<float>({
        { 11, 12, 13, 14, 15, 16, 17, 18},
        { 21, 22, 23, 24, 25, 26, 27, 28},
        { 31, 32, 33, 34, 35, 36, 37, 38},
    });

    EXPECT_DEATH(matrix<float>::view(data, 2, 3, 0, 7), "Assertion failed.*");
    EXPECT_DEATH(matrix<float>::view(data, 3, 4, 0, 7), "Assertion failed.*");
}

TEST(squashmax_test_case, squashmax_test)
{
    auto data = make_matrix<float>({
        { 0.1, 0.2, 0.3, 0.21, 0.11, 0.03 }
    });

    matrix<float> squashed = squashmax<float>(data);

    EXPECT_NEAR(squashed(0, 0), 0, 0.01);
    EXPECT_NEAR(squashed(0, 1), 0, 0.01);
    EXPECT_NEAR(squashed(0, 2), 1, 0.01);
    EXPECT_NEAR(squashed(0, 3), 0, 0.01);
    EXPECT_NEAR(squashed(0, 4), 0, 0.01);
    EXPECT_NEAR(squashed(0, 5), 0, 0.01);
}

TEST(reduce_column_sum, simple_reduce_column_sum)
{
    auto data = make_matrix<float>({
        {0, 0, 0},
        {0, 0, 1}, /* 1 */
        {0, 1, 0}, /* 2 */
        {0, 1, 1}, /* 3 */
        {1, 0, 0}, /* 4 */
        {1, 0, 1}, /* 5 */
        {1, 1, 0}, /* 6 */
        {1, 1, 1}  /* 7 */
    });

    auto data2 = data.reduce_column_sum();

    EXPECT_EQ(data2.size1(), 1);
    EXPECT_EQ(data2.size2(), 3);

    EXPECT_NEAR(data2(0, 0), 4, 0.01);
    EXPECT_NEAR(data2(0, 1), 4, 0.01);
    EXPECT_NEAR(data2(0, 2), 4, 0.01);
}

// TEST(columnwise_add_test, columnwise_basic_test)
// {
//     auto data = make_matrix<float>({
//         {0, 0, 0},
//         {0, 0, 1}, /* 1 */
//         {0, 1, 0}, /* 2 */
//         {0, 1, 1}, /* 3 */
//         {1, 0, 0}, /* 4 */
//         {1, 0, 1}, /* 5 */
//         {1, 1, 0}, /* 6 */
//         {1, 1, 1}  /* 7 */
//     });
//
//     auto to_add = make_matrix<float>({
//         { 1, 2, 3 },
//         { 4, 5, 6 }
//     });
//
//     data.columnwise_add(to_add);
//
//     EXPECT_NEAR(data(0, 0), 1, 0.01);
//     EXPECT_NEAR(data(1, 0), 1, 0.01);
//     EXPECT_NEAR(data(2, 0), 1, 0.01);
//     EXPECT_NEAR(data(3, 0), 1, 0.01);
//     EXPECT_NEAR(data(4, 0), 2, 0.01);
//     EXPECT_NEAR(data(5, 0), 2, 0.01);
//     EXPECT_NEAR(data(6, 0), 2, 0.01);
//     EXPECT_NEAR(data(7, 0), 2, 0.01);
//
//     EXPECT_NEAR(data(0, 1), 2, 0.01);
//     EXPECT_NEAR(data(1, 1), 2, 0.01);
//     EXPECT_NEAR(data(2, 1), 3, 0.01);
//     EXPECT_NEAR(data(3, 1), 3, 0.01);
//     EXPECT_NEAR(data(4, 1), 2, 0.01);
//     EXPECT_NEAR(data(5, 1), 2, 0.01);
//     EXPECT_NEAR(data(6, 1), 3, 0.01);
//     EXPECT_NEAR(data(7, 1), 3, 0.01);
//
//     EXPECT_NEAR(data(0, 2), 3, 0.01);
//     EXPECT_NEAR(data(1, 2), 4, 0.01);
//     EXPECT_NEAR(data(2, 2), 3, 0.01);
//     EXPECT_NEAR(data(3, 2), 4, 0.01);
//     EXPECT_NEAR(data(4, 2), 3, 0.01);
//     EXPECT_NEAR(data(5, 2), 4, 0.01);
//     EXPECT_NEAR(data(6, 2), 3, 0.01);
//     EXPECT_NEAR(data(7, 2), 4, 0.01);
// }

// TEST(adagrad_update, adagrad_update_1)
// {
//     auto memo = make_matrix<float>({
//         { 7.4246e-07, 0.000332555, 2.49854e-06, 4.52723e-09, 1.74763e-12, 6.36695e-14, 5.83191e-14, 1.60086e-05 },
//         { 0, 0, 0, 0, 0, 0, 0, 0 },
//         { 0, 0, 0, 0, 0, 0, 0, 0 },
//         { 0.000111749, 0.0503772, 0.000174712, 1.76435e-06, 6.41502e-09, 9.06837e-10, 2.15051e-10, 0.00237573 }
//     });
//
//     auto weight_deltas = make_matrix<float>({
//
//     });
// }

