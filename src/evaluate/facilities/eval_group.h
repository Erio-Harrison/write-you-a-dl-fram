#pragma once

// Include the header file for evaluation units.
#include <evaluate/facilities/eval_unit.h>
#include <list>
#include <memory>

// The `BaseEvalGroup` class template is an abstract base class for evaluation groups.
// It provides a common interface for different types of evaluation groups that work with a specific device type.
template <typename TDevice>
class BaseEvalGroup
{
public:
    // Virtual destructor to ensure proper cleanup when derived classes are deleted through a base class pointer.
    virtual ~BaseEvalGroup() = default;

    // Pure virtual function to get an evaluation unit from the group.
    // Derived classes must implement this function to provide a way to retrieve an evaluation unit.
    virtual std::shared_ptr<BaseEvalUnit<TDevice>> GetEvalUnit() = 0;

    // Pure virtual function to merge an lvalue reference of a base evaluation unit into the group.
    // Derived classes must implement this function to handle the merging operation.
    virtual void Merge(BaseEvalUnit<TDevice>&) = 0;

    // Pure virtual function to merge an rvalue reference of a base evaluation unit into the group.
    // Derived classes must implement this function to handle the merging operation for temporary objects.
    virtual void Merge(BaseEvalUnit<TDevice>&&) = 0;
};

// The `TrivalEvalGroup` class template is a derived class from `BaseEvalGroup`.
// It provides a simple implementation for managing evaluation units of a specific type.
template <typename TEvalUnit>
class TrivalEvalGroup : public BaseEvalGroup<typename TEvalUnit::DeviceType>
{
    // Alias for the device type used by the evaluation unit.
    using DeviceType = typename TEvalUnit::DeviceType;
public:
    // Overrides the `GetEvalUnit` function from the base class.
    // It retrieves an evaluation unit from the list if available.
    std::shared_ptr<BaseEvalUnit<DeviceType>> GetEvalUnit() override
    {
        // A shared pointer to hold the result evaluation unit.
        std::shared_ptr<BaseEvalUnit<DeviceType>> res;
        // Check if the list of evaluation units is not empty.
        if (!m_unitList.empty())
        {
            // Create a new shared pointer to a `TEvalUnit` object by moving the front element of the list.
            res = std::make_shared<TEvalUnit>(std::move(m_unitList.front()));
            // Remove the front element from the list.
            m_unitList.pop_front();
        }
        // Return the result shared pointer.
        return res;
    }

    // Overrides the `Merge` function for lvalue references from the base class.
    // It adds the given evaluation unit to the list.
    void Merge(BaseEvalUnit<DeviceType>& unit) override
    {
        // Cast the base evaluation unit to the specific evaluation unit type and add it to the list.
        m_unitList.push_back(static_cast<TEvalUnit&>(unit));
    }

    // Overrides the `Merge` function for rvalue references from the base class.
    // It adds the given temporary evaluation unit to the list.
    void Merge(BaseEvalUnit<DeviceType>&& unit) override
    {
        // Cast the rvalue base evaluation unit to the specific evaluation unit type and add it to the list.
        m_unitList.push_back(static_cast<TEvalUnit&&>(unit));
    }

private:
    // A list to store evaluation units of type `TEvalUnit`.
    std::list<TEvalUnit> m_unitList;
};