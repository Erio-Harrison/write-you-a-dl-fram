#pragma once

#include <unordered_map>
#include <data/facilities/tags.h>
#include <vector>

/**
 * @brief A base class template for evaluation units that work with a specific device type.
 * 
 * This class serves as an abstract base for all evaluation units that are designed to operate
 * on a particular device type. It defines the common interface that derived evaluation units
 * must implement.
 * 
 * @tparam TDevice The type of the device on which the evaluation unit will operate.
 */
template <typename TDevice>
class BaseEvalUnit
{
public:
    /**
     * @brief An alias for the device type used by this evaluation unit.
     * 
     * This alias makes it easier to refer to the device type within the class and in other
     * parts of the code that interact with this evaluation unit.
     */
    using DeviceType = TDevice;

    /**
     * @brief Virtual destructor.
     * 
     * The virtual destructor ensures that when a derived class object is deleted through a
     * pointer to the base class, the correct derived class destructor is called.
     * It is set to default, which means the compiler will generate a default implementation.
     */
    virtual ~BaseEvalUnit() = default;

    /**
     * @brief Pure virtual function to perform the evaluation.
     * 
     * This function must be implemented by all derived classes. It is responsible for
     * carrying out the actual evaluation task specific to the derived evaluation unit.
     */
    virtual void Eval() = 0;
};