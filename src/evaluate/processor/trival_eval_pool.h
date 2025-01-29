#pragma once

#include <data/facilities/tags.h>
#include <evaluate/facilities/eval_pool.h>

// Specialization of TrivalEvalPool for CPU
template <> 
class TrivalEvalPool<DeviceTags::CPU> : public BaseEvalPool<DeviceTags::CPU> {
public:
    // Singleton instance method for the CPU version of TrivalEvalPool
    static TrivalEvalPool& Instance() {
        static TrivalEvalPool inst; // static ensures it's a singleton
        return inst;
    }

private:
    // Private constructor to prevent instantiation from outside the class
    TrivalEvalPool() = default;

public:
    // Process method for CPU: This will evaluate the evaluation unit
    void Process(std::shared_ptr<BaseEvalUnit<DeviceTags::CPU>>& eu) override {
        eu->Eval();  // Call Eval on the CPU evaluation unit
    }

    // Barrier method: No-op for CPU as we don't need synchronization here
    void Barrier() override {}
};

// Specialization of TrivalEvalPool for GPU
template <> 
class TrivalEvalPool<DeviceTags::GPU> : public BaseEvalPool<DeviceTags::GPU> {
public:
    // Singleton instance method for the GPU version of TrivalEvalPool
    static TrivalEvalPool& Instance() {
        static TrivalEvalPool inst; // static ensures it's a singleton
        return inst;
    }

private:
    // Private constructor to prevent instantiation from outside the class
    TrivalEvalPool() = default;

public:
    // Process method for GPU: This will evaluate the evaluation unit
    void Process(std::shared_ptr<BaseEvalUnit<DeviceTags::GPU>>& eu) override {
        eu->Eval();  // Call Eval on the GPU evaluation unit
    }

    // Barrier method: Sync the GPU if necessary (e.g., cudaDeviceSynchronize or hipDeviceSynchronize)
    void Barrier() override {
        // If needed, implement GPU synchronization logic here
        // Example: cudaDeviceSynchronize() or hipDeviceSynchronize()
    }
};
