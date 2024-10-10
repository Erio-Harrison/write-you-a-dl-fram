#include <gtest/gtest.h>
#include "../src/tensor/tensor_advanced.hpp"

TEST(AdvancedTensorTest, BroadcastAdd) {
    AdvancedTensor<float, 2> t1(std::array<size_t, 2>{2, 3});
    AdvancedTensor<float, 1> t2(std::array<size_t, 1>{3});

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            t1({{i, j}}) = i * 3 + j + 1;
        }
    }
    for (size_t i = 0; i < 3; ++i) {
        t2({{i}}) = i + 1;
    }

    auto result = t1.broadcast_add(t2);

    ASSERT_EQ(result.shape()[0], 2);
    ASSERT_EQ(result.shape()[1], 3);

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(result({{i, j}}), t1({{i, j}}) + t2({{j}}));
        }
    }
}

TEST(AdvancedTensorTest, OptimizeAdd) {
    AdvancedTensor<float, 1> t1(std::array<size_t, 1>{1000});
    AdvancedTensor<float, 1> t2(std::array<size_t, 1>{1000});

    for (size_t i = 0; i < 1000; ++i) {
        t1({{i}}) = i;
        t2({{i}}) = 1000 - i;
    }

    t1.optimize_add(t2);

    for (size_t i = 0; i < 1000; ++i) {
        EXPECT_FLOAT_EQ(t1({{i}}), 1000);
    }
}

TEST(AdvancedTensorTest, OptimizeSub) {
    AdvancedTensor<float, 1> t1(std::array<size_t, 1>{1000});
    AdvancedTensor<float, 1> t2(std::array<size_t, 1>{1000});

    for (size_t i = 0; i < 1000; ++i) {
        t1({{i}}) = 2000 - i;
        t2({{i}}) = i;
    }

    t1.optimize_sub(t2);

    for (size_t i = 0; i < 1000; ++i) {
        EXPECT_FLOAT_EQ(t1({{i}}), 2000 - 2*i);
    }
}

TEST(AdvancedTensorTest, OptimizeMul) {
    AdvancedTensor<float, 1> t1(std::array<size_t, 1>{1000});
    AdvancedTensor<float, 1> t2(std::array<size_t, 1>{1000});

    for (size_t i = 0; i < 1000; ++i) {
        t1({{i}}) = i + 1;
        t2({{i}}) = 2;
    }

    t1.optimize_mul(t2);

    for (size_t i = 0; i < 1000; ++i) {
        EXPECT_FLOAT_EQ(t1({{i}}), 2*(i + 1));
    }
}

TEST(AdvancedTensorTest, OptimizeDiv) {
    AdvancedTensor<float, 1> t1(std::array<size_t, 1>{1000});
    AdvancedTensor<float, 1> t2(std::array<size_t, 1>{1000});

    for (size_t i = 0; i < 1000; ++i) {
        t1({{i}}) = 2 * (i + 1);
        t2({{i}}) = 2;
    }

    t1.optimize_div(t2);

    for (size_t i = 0; i < 1000; ++i) {
        EXPECT_FLOAT_EQ(t1({{i}}), i + 1);
    }
}

TEST(AdvancedTensorTest, DivideByZero) {
    AdvancedTensor<float, 1> t1(std::array<size_t, 1>{10});
    AdvancedTensor<float, 1> t2(std::array<size_t, 1>{10});

    for (size_t i = 0; i < 10; ++i) {
        t1({{i}}) = i;
        t2({{i}}) = 0;
    }

    EXPECT_THROW(t1.optimize_div(t2), std::runtime_error);
}


TEST(AdvancedTensorTest, DifferentShapes) {
    AdvancedTensor<float, 2> t1(std::array<size_t, 2>{2, 3});
    AdvancedTensor<float, 2> t2(std::array<size_t, 2>{3, 2});

    EXPECT_THROW(t1.optimize_add(t2), std::invalid_argument);
    EXPECT_THROW(t1.optimize_sub(t2), std::invalid_argument);
    EXPECT_THROW(t1.optimize_mul(t2), std::invalid_argument);
    EXPECT_THROW(t1.optimize_div(t2), std::invalid_argument);
}