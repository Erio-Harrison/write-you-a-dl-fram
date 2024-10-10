#include <gtest/gtest.h>
#include "../src/tensor/tensor_multithreading.hpp"

TEST(MultithreadedTensorTest, ParallelAdd) {
    MultithreadedTensor<float, 1> t1(std::array<size_t, 1>{{10000}});
    MultithreadedTensor<float, 1> t2(std::array<size_t, 1>{{10000}});


    for (size_t i = 0; i < 10000; ++i) {
        t1({{i}}) = i;
        t2({{i}}) = 10000 - i;
    }

    t1.parallel_add(t2);

    for (size_t i = 0; i < 10000; ++i) {
        EXPECT_FLOAT_EQ(t1({{i}}), 10000);
    }
}

TEST(MultithreadedTensorTest, ParallelBroadcastAdd) {
    MultithreadedTensor<float, 2> t1(std::array<size_t, 2>{{100, 100}});
    MultithreadedTensor<float, 1> t2(std::array<size_t, 1>{{100}});

    for (size_t i = 0; i < 100; ++i) {
        for (size_t j = 0; j < 100; ++j) {
            t1({{i, j}}) = i * 100 + j;
        }
        t2({{i}}) = i;
    }

    auto result = t1.parallel_broadcast_add(t2);

    for (size_t i = 0; i < 100; ++i) {
        for (size_t j = 0; j < 100; ++j) {
            EXPECT_FLOAT_EQ(result({{i, j}}), t1({{i, j}}) + t2({{j}}));
        }
    }
}

TEST(MultithreadedTensorTest, ParallelSum) {
    const size_t size = 10000;
    MultithreadedTensor<double, 1> t(std::array<size_t, 1>{{size}});

    double expected_sum = 0.0;
    for (size_t i = 0; i < size; ++i) {
        t({{i}}) = static_cast<double>(i);
        expected_sum += static_cast<double>(i);
    }

    double sum = t.parallel_sum();

    EXPECT_NEAR(sum, expected_sum, 1e-10 * expected_sum);
}
