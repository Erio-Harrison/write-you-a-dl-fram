#pragma once

#include <evaluate/facilities/eval_unit.h>
#include <memory>

/**
 * @brief Enumeration class representing different types of evaluation pools.
 * 
 * This enum class defines the available types of evaluation pools. Currently,
 * only one type, `Trival`, is defined.
 */
enum class EvalPoolEnum
{
    // Represents a trivial evaluation pool.
    Trival
};

/**
 * @brief Base class template for evaluation pools.
 * 
 * This class serves as an abstract base for all evaluation pools that work with a specific device type.
 * It defines the common interface that derived evaluation pools must implement.
 * 
 * @tparam TDevice The type of the device on which the evaluation pool will operate.
 */
template <typename TDevice>
class BaseEvalPool
{
public:
    /**
     * @brief Virtual destructor.
     * 
     * Ensures that the correct destructor of derived classes is called when deleting through a base class pointer.
     */
    virtual ~BaseEvalPool() = default;

    /**
     * @brief Pure virtual function to process an evaluation unit.
     * 
     * Derived classes must implement this function to handle the processing of evaluation units.
     * 
     * @param unit A shared pointer to a base evaluation unit.
     */
    virtual void Process(std::shared_ptr<BaseEvalUnit<TDevice>>& unit) = 0;

    /**
     * @brief Pure virtual function to synchronize the evaluation pool.
     * 
     * Derived classes must implement this function to ensure that all evaluation tasks are completed
     * before proceeding further.
     */
    virtual void Barrier() = 0;
};

/**
 * @brief Forward declaration of the TrivalEvalPool class template.
 * 
 * This forward declaration is used to declare the `TrivalEvalPool` class template without providing
 * its full definition. It allows other parts of the code to use pointers or references to `TrivalEvalPool`
 * objects before its actual implementation is available.
 * 
 * @tparam TDevice The type of the device on which the trivial evaluation pool will operate.
 */
template <typename TDevice>
class TrivalEvalPool;