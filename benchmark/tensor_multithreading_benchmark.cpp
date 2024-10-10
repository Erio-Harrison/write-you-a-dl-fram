#include <benchmark/benchmark.h>
#include "../src/tensor/tensor_multithreading.hpp"

template <typename T>
static void BM_ParallelAdd(benchmark::State& state) {
    const size_t size = state.range(0);
    std::array<size_t, 1> shape = {size};
    MultithreadedTensor<T, 1> t1(shape);
    MultithreadedTensor<T, 1> t2(shape);

    for (size_t i = 0; i < size; ++i) {
        t1({{i}}) = static_cast<T>(i);
        t2({{i}}) = static_cast<T>(size - i);
    }

    for (auto _ : state) {
        t1.parallel_add(t2);
        benchmark::DoNotOptimize(t1);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * size);
    state.SetBytesProcessed(state.iterations() * size * sizeof(T) * 2);
}

BENCHMARK_TEMPLATE(BM_ParallelAdd, float)->Range(1<<10, 1<<20);
BENCHMARK_TEMPLATE(BM_ParallelAdd, double)->Range(1<<10, 1<<20);

template <typename T>
static void BM_ParallelSum(benchmark::State& state) {
    const size_t size = state.range(0);
    std::array<size_t, 1> shape = {size};
    MultithreadedTensor<T, 1> t(shape);

    for (size_t i = 0; i < size; ++i) {
        t({{i}}) = static_cast<T>(i);
    }

    for (auto _ : state) {
        T sum = t.parallel_sum();
        benchmark::DoNotOptimize(sum);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * size);
    state.SetBytesProcessed(state.iterations() * size * sizeof(T));
}

BENCHMARK_TEMPLATE(BM_ParallelSum, float)->Range(1<<10, 1<<20);
BENCHMARK_TEMPLATE(BM_ParallelSum, double)->Range(1<<10, 1<<20);

static void BM_ParallelBroadcastAdd(benchmark::State& state) {
    const size_t rows = state.range(0);
    const size_t cols = state.range(1);
    std::array<size_t, 2> shape2d = {rows, cols};
    std::array<size_t, 1> shape1d = {cols};
    MultithreadedTensor<float, 2> t1(shape2d);
    MultithreadedTensor<float, 1> t2(shape1d);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            t1({{i, j}}) = static_cast<float>(i * cols + j);
        }
    }
    for (size_t j = 0; j < cols; ++j) {
        t2({{j}}) = static_cast<float>(j);
    }

    for (auto _ : state) {
        auto result = t1.parallel_broadcast_add(t2);
        benchmark::DoNotOptimize(result);
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * rows * cols);
    state.SetBytesProcessed(state.iterations() * rows * cols * sizeof(float));
}

BENCHMARK(BM_ParallelBroadcastAdd)
    ->Args({1<<10, 1<<10})
    ->Args({1<<12, 1<<8})
    ->Args({1<<8, 1<<12});

BENCHMARK_MAIN();