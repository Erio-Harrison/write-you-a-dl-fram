#pragma once

#include <memory>
// Include the header file related to evaluation handles.
#include <evaluate/facilities/eval_handle.h>

// The `EvalBuffer` class template is designed to manage an evaluation buffer for a specific data type.
// It provides methods to access evaluation handles and check the evaluation status.
template <typename TData>
class EvalBuffer
{
public:
    // An alias for the data type that this evaluation buffer manages.
    using DataType = TData;

    // This method returns the evaluation handle associated with the buffer.
    // It allows external code to access and manipulate the evaluation state and data.
    auto Handle() const
    {
        return m_handle;
    }

    // This method returns a const evaluation handle for the evaluation handle of the buffer.
    // It provides a read - only way to access the data and evaluation state.
    auto ConstHandle() const
    {
        return ConstEvalHandle<EvalHandle<TData>>(m_handle);
    }

    // This method checks if the data in the buffer has been evaluated.
    // It returns a boolean value indicating the evaluation status and is declared as noexcept
    // to ensure it won't throw any exceptions.
    bool IsEvaluated() const noexcept
    {
        return m_handle.IsEvaluated();
    }

private:
    // The evaluation handle that stores and manages the data and its evaluation status.
    EvalHandle<TData> m_handle;
};