#include <gtest/gtest.h>
#include "../src/autograd/autograd.hpp"

TEST(AutogradTest, Addition) {
    AdvancedTensor<float, 1> data1(std::array<size_t, 1>{{1}});
    AdvancedTensor<float, 1> data2(std::array<size_t, 1>{{1}});
    data1({{0}}) = 2;
    data2({{0}}) = 3;

    Variable<float, 1> var1(data1);
    Variable<float, 1> var2(data2);

    auto result = var1 + var2;
    result.backward();

    EXPECT_FLOAT_EQ(result.data()({{0}}), 5);
    EXPECT_FLOAT_EQ(var1.grad()({{0}}), 1);
    EXPECT_FLOAT_EQ(var2.grad()({{0}}), 1);
}

TEST(AutogradTest, Multiplication) {
    AdvancedTensor<float, 1> data1(std::array<size_t, 1>{{1}});
    AdvancedTensor<float, 1> data2(std::array<size_t, 1>{{1}});
    data1({{0}}) = 2;
    data2({{0}}) = 3;

    Variable<float, 1> var1(data1);
    Variable<float, 1> var2(data2);

    auto result = var1 * var2;
    result.backward();

    EXPECT_FLOAT_EQ(result.data()({{0}}), 6);
    EXPECT_FLOAT_EQ(var1.grad()({{0}}), 3);
    EXPECT_FLOAT_EQ(var2.grad()({{0}}), 2);
}

TEST(AutogradTest, ComplexComputation) {
    AdvancedTensor<float, 1> data1(std::array<size_t, 1>{{1}});
    AdvancedTensor<float, 1> data2(std::array<size_t, 1>{{1}});
    AdvancedTensor<float, 1> data3(std::array<size_t, 1>{{1}});
    data1({{0}}) = 2;
    data2({{0}}) = 3;
    data3({{0}}) = 4;

    Variable<float, 1> var1(data1);
    Variable<float, 1> var2(data2);
    Variable<float, 1> var3(data3);

    auto temp = var1 * var2;
    auto result = temp + var3;
    result.backward();

    EXPECT_FLOAT_EQ(result.data()({{0}}), 10);
    EXPECT_FLOAT_EQ(var1.grad()({{0}}), 3);
    EXPECT_FLOAT_EQ(var2.grad()({{0}}), 2);
    EXPECT_FLOAT_EQ(var3.grad()({{0}}), 1);
}
