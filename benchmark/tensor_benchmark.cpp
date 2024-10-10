#include <benchmark/benchmark.h>
#include "../src/tensor/tensor_advanced.hpp"

static void BM_OptimizeAdd(benchmark::State& state) {
    const size_t size = 10000;
    AdvancedTensor<float, 1> t1({size});
    AdvancedTensor<float, 1> t2({size});

    for (size_t i = 0; i < size; ++i) {
        t1({{i}}) = static_cast<float>(i);
        t2({{i}}) = static_cast<float>(size - i);
    }

    for (auto _ : state) {
        t1.optimize_add(t2);
        benchmark::DoNotOptimize(t1);
        benchmark::ClobberMemory();
    }
}

BENCHMARK(BM_OptimizeAdd);

static void BM_NormalAdd(benchmark::State& state) {
    const size_t size = 10000;
    AdvancedTensor<float, 1> t1({size});
    AdvancedTensor<float, 1> t2({size});

    for (size_t i = 0; i < size; ++i) {
        t1({{i}}) = static_cast<float>(i);
        t2({{i}}) = static_cast<float>(size - i);
    }

    for (auto _ : state) {
        for (size_t i = 0; i < size; ++i) {
            t1({{i}}) += t2({{i}});
        }
        benchmark::DoNotOptimize(t1);
        benchmark::ClobberMemory();
    }
}

BENCHMARK(BM_NormalAdd);

template <size_t Size>
static void BM_OptimizeAdd_Size(benchmark::State& state) {
    AdvancedTensor<float, 1> t1({Size});
    AdvancedTensor<float, 1> t2({Size});

    for (size_t i = 0; i < Size; ++i) {
        t1({{i}}) = static_cast<float>(i);
        t2({{i}}) = static_cast<float>(Size - i);
    }

    for (auto _ : state) {
        t1.optimize_add(t2);
        benchmark::DoNotOptimize(t1);
        benchmark::ClobberMemory();
    }
}

BENCHMARK_TEMPLATE(BM_OptimizeAdd_Size, 100);
BENCHMARK_TEMPLATE(BM_OptimizeAdd_Size, 1000);
BENCHMARK_TEMPLATE(BM_OptimizeAdd_Size, 10000);
BENCHMARK_TEMPLATE(BM_OptimizeAdd_Size, 100000);

BENCHMARK_MAIN();