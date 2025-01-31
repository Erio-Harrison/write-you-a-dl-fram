#pragma once

// Include necessary headers related to evaluation processing, groups, handles, pools, and units.
#include <evaluate/processor/trival_eval_pool.h>
#include <evaluate/facilities/eval_group.h>
#include <evaluate/facilities/eval_handle.h>
#include <evaluate/facilities/eval_pool.h>
#include <evaluate/facilities/eval_unit.h>

// Include standard library headers for various data structures and utility functions.
#include <vector>
#include <cassert>
#include <list>
#include <memory>
#include <stdexcept>
#include <unordered_set>
#include <typeindex>
#include <algorithm>

// Namespace for evaluation planning related functions and types.
namespace NSEvalPlan
{
    /**
     * @brief Calculate the operand depth based on the dependency map and parameter pointers.
     * @param depMap An unordered map that stores the depth of each operand.
     * @param paramPtr A vector of pointers to the operands.
     * @return The maximum depth among the operands, or -1 if no operand is found in the map.
     */
    inline size_t OperandDepth(const std::unordered_map<const void*, size_t>& depMap,
                               const std::vector<const void*>& paramPtr)
    {
        // Initialize the result to -1, indicating no valid depth found yet.
        int res = -1;
        
        // Iterate through each parameter pointer.
        for (auto p : paramPtr)
        {
            // Find the parameter pointer in the dependency map.
            auto it = depMap.find(p);
            // If the pointer is found in the map, update the result with the maximum depth.
            if (it != depMap.end()) res = std::max(res, (int)(it->second));
        }

        // Return the calculated depth as a size_t type.
        return (size_t)res;
    }
}

// Define an alias for an evaluation cluster, which is an unordered map of evaluation groups.
// The key is a type index, and the value is a shared pointer to a base evaluation group.
template <typename TDevice>
using EvalCluster = std::unordered_map<std::type_index, std::shared_ptr<BaseEvalGroup<TDevice>>>;

// Class representing an evaluation layer, which manages a sequence of evaluation clusters.
template <typename TDevice>
class EvalLayer
{
public:
    /**
     * @brief Get the number of evaluation clusters in the layer.
     * @return The size of the evaluation sequence.
     */
    size_t Size() const
    {
        return m_evalSeq.size();
    }

    /**
     * @brief Overload the [] operator to access an evaluation cluster at a specific index.
     * @param i The index of the evaluation cluster.
     * @return A reference to the evaluation cluster at the given index.
     */
    EvalCluster<TDevice>& operator[] (size_t i)
    {
        return m_evalSeq[i];
    }

    /**
     * @brief Check if the evaluation layer is empty.
     * @return true if the evaluation sequence is empty, false otherwise.
     */
    bool Empty() const
    {
        return m_evalSeq.empty();
    }

    /**
     * @brief Clear all data in the evaluation layer, including the evaluation sequence,
     *        operands, and outputs.
     */
    void Clear()
    {
        m_evalSeq.clear();
        m_operands.clear();
        m_outputs.clear();
    }

    /**
     * @brief Register an evaluation request in the layer.
     * @tparam TEvalGroup The type of the evaluation group.
     * @tparam TEvalUnit The type of the evaluation unit.
     * @param evalReq The evaluation request to be registered.
     * @param resPtr A pointer to the result of the evaluation.
     * @param paramPtr A vector of pointers to the parameters of the evaluation.
     */
    template <typename TEvalGroup, typename TEvalUnit>
    void EvalRegister(TEvalUnit&& evalReq, const void* resPtr,
                      const std::vector<const void*>& paramPtr)
    {
        // If the result pointer is null, do nothing.
        if (!resPtr) return;
        // If the result pointer is already in the outputs map, do nothing.
        if (m_outputs.find(resPtr) != m_outputs.end()) return;

        // Calculate the depth of the evaluation based on the operand depths.
        size_t depth = NSEvalPlan::OperandDepth(m_outputs, paramPtr) + 1;

        // Resize the evaluation sequence if necessary to accommodate the new depth.
        if (m_evalSeq.size() <= (size_t)depth)
        {
            m_evalSeq.resize(depth + 1);
        }
        // Get the evaluation cluster at the calculated depth.
        EvalCluster<TDevice>& ec = m_evalSeq[depth];

        // Get the type index of the evaluation group.
        const auto typeIndex = std::type_index(typeid(TEvalGroup));
        // Find the evaluation group in the cluster.
        auto it = ec.find(typeIndex);

        // If the evaluation group is not found, create a new one and insert it into the cluster.
        if (it == ec.end())
        {
            it = ec.insert({typeIndex, std::make_shared<TEvalGroup>()}).first;
        }
        // Merge the evaluation request into the evaluation group.
        it->second->Merge(std::forward<TEvalUnit>(evalReq));

        // Insert the result pointer and its depth into the outputs map.
        m_outputs.insert({resPtr, depth});
    }

private:
    // A vector of evaluation clusters representing the evaluation sequence.
    std::vector<EvalCluster<TDevice>> m_evalSeq;
    // An unordered set of pointers to the operands.
    std::unordered_set<const void*> m_operands;
    // An unordered map that stores the depth of each output.
    std::unordered_map<const void*, size_t> m_outputs;
};

// Class representing an evaluation plan, which manages evaluation layers and the evaluation pool.
template <typename TDevice>
class EvalPlan
{
private:
    /**
     * @brief Get a reference to the global evaluation pool enumeration.
     * @return A reference to the global evaluation pool enumeration.
     */
    static EvalPoolEnum& GlobalEvalPool()
    {
        // Initialize the global evaluation pool to the trivial type.
        static EvalPoolEnum inst = EvalPoolEnum::Trival;
        return inst;
    }
    
    /**
     * @brief Get a reference to the thread-local evaluation pool enumeration.
     * @return A reference to the thread-local evaluation pool enumeration.
     */
    static EvalPoolEnum& ThreadEvalPool()
    {
        // Initialize the thread-local evaluation pool to the global evaluation pool.
        static thread_local EvalPoolEnum inst = GlobalEvalPool();
        return inst;
    }
    
    /**
     * @brief Get a reference to the thread-local evaluation plan instance.
     * @return A reference to the thread-local evaluation plan instance.
     */
    static EvalPlan& ThreadInst()
    {
        // Initialize the thread-local evaluation plan instance.
        static thread_local EvalPlan inst;
        return inst;
    }

public:
    /**
     * @brief Set the global evaluation pool type.
     * @param epType The type of the evaluation pool to set.
     */
    static void SetEvalPool(EvalPoolEnum epType)
    {
        GlobalEvalPool() = epType;
    }

    /**
     * @brief Register an evaluation request in the evaluation plan.
     * @tparam TEvalGroup The type of the evaluation group.
     * @tparam TEvalUnit The type of the evaluation unit.
     * @param evalReq The evaluation request to be registered.
     * @param outputPtr A pointer to the output of the evaluation.
     * @param paramPtr A vector of pointers to the parameters of the evaluation.
     */
    template <typename TEvalGroup, typename TEvalUnit>
    static void Register(TEvalUnit&& evalReq, const void* outputPtr,
                         const std::vector<const void*>& paramPtr)
    {
        // Forward the registration to the thread-local evaluation plan instance.
        ThreadInst().template EvalRegister<TEvalGroup>(std::forward<TEvalUnit>(evalReq), outputPtr, paramPtr);
    }

    /**
     * @brief Perform the evaluation according to the evaluation plan.
     */
    static void Eval()
    {
        // Get the thread-local evaluation plan instance.
        EvalPlan& plan = ThreadInst();
        // Check if the thread-local evaluation pool needs to be updated or if it is null.
        if ((ThreadEvalPool() != GlobalEvalPool()) || (!plan.m_evalPool))
        {
            // Select the appropriate evaluation pool based on the global evaluation pool type.
            switch(GlobalEvalPool())
            {
            case EvalPoolEnum::Trival:
                plan.m_evalPool = &(TrivalEvalPool<TDevice>::Instance());
                break;
            default:
                // Assert false if an unsupported evaluation pool type is encountered.
                assert(false);
            }
            // Update the thread-local evaluation pool to match the global one.
            ThreadEvalPool() = GlobalEvalPool();
        }
        // Throw an exception if no evaluation pool is available.
        if (!plan.m_evalPool)
        {
            throw std::runtime_error("No Evaluation Pool is available.");
        }
        
        // Perform the layer-wise evaluation.
        plan.DoLayerEval();
    }

private:
    /**
     * @brief Constructor for the evaluation plan. Initializes the evaluation pool pointer to null
     *        and resizes the evaluation layers list.
     */
    EvalPlan()
        : m_evalPool(nullptr)
    {
        m_evalLayers.resize(1);
    }

    /**
     * @brief Register an evaluation request in the current evaluation layer.
     * @tparam TEvalGroup The type of the evaluation group.
     * @tparam TEvalUnit The type of the evaluation unit.
     * @param evalReq The evaluation request to be registered.
     * @param outputPtr A pointer to the output of the evaluation.
     * @param paramPtr A vector of pointers to the parameters of the evaluation.
     */
    template <typename TEvalGroup, typename TEvalUnit>
    void EvalRegister(TEvalUnit&& evalReq, const void* outputPtr,
                      const std::vector<const void*>& paramPtr)
    {
        // Get the current evaluation layer.
        auto& curLayer = m_evalLayers.back();
        // Forward the registration to the current evaluation layer.
        curLayer.template EvalRegister<TEvalGroup>(std::forward<TEvalUnit>(evalReq),
                                                   outputPtr, paramPtr);
    }

    /**
     * @brief Perform the layer-wise evaluation.
     */
    void DoLayerEval()
    {
        // Get the current evaluation layer.
        EvalLayer<TDevice>& curLayer = m_evalLayers.back();
        // If the current layer is empty, do nothing.
        if (curLayer.Empty()) return;

        // Push a new empty evaluation layer onto the list.
        m_evalLayers.push_back(EvalLayer<TDevice>{});
        // Get the length of the evaluation sequence in the current layer.
        size_t seqLen = curLayer.Size();
        // Iterate through each evaluation cluster in the current layer.
        for (size_t i = 0; i < seqLen; ++i)
        {
            // Get the evaluation cluster at the current index.
            EvalCluster<TDevice>& ec = curLayer[i];
            // Iterate through each evaluation group in the cluster.
            for (auto& eg : ec)
            {
                // Get evaluation units from the group and process them until there are none left.
                while(auto unit = eg.second->GetEvalUnit())
                {
                    m_evalPool->Process(unit);
                }
            }
            // Synchronize the evaluation pool after processing all units in the cluster.
            m_evalPool->Barrier();
            // If the new layer is not empty, recursively perform layer-wise evaluation.
            if (!m_evalLayers.back().Empty())
            {
                DoLayerEval();
            }
        }
        // Pop the new layer from the list.
        m_evalLayers.pop_back();
        // Clear the current layer.
        curLayer.Clear();
    }

private:
    // A list of evaluation layers.
    std::list<EvalLayer<TDevice>> m_evalLayers;
    // A pointer to the base evaluation pool.
    BaseEvalPool<TDevice>* m_evalPool;
};

/**
 * @brief Evaluate the given data using the evaluation plan.
 * @tparam TData The type of the data to be evaluated.
 * @param data The data to be evaluated.
 * @return The evaluated data.
 */
template <typename TData>
auto Evaluate(const TData& data)
{
    // Get the device type of the data.
    using DeviceType = typename TData::DeviceType;
    // Register the evaluation and get the evaluation handle.
    auto evalHandle = data.EvalRegister();
    // Perform the evaluation using the evaluation plan.
    EvalPlan<DeviceType>::Eval();
    // Return the evaluated data from the evaluation handle.
    return evalHandle.Data();
}