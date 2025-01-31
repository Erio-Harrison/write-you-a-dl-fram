#pragma once

#include <cassert>
#include <memory>
#include <stdexcept>

// The `EvalHandle` class template is designed to manage data with an evaluation flag.
// It provides methods to access, modify, and mark the data as evaluated.
template <typename TData>
class EvalHandle
{
    // `DataWithEvalInfo` is a nested struct that holds the actual data and an evaluation flag.
    struct DataWithEvalInfo
    {
        TData m_data;
        bool m_eval = false;
    };

public:
    // Default constructor. Initializes the `m_data` member with a shared pointer to a `DataWithEvalInfo` object.
    EvalHandle()
        : m_data(std::make_shared<DataWithEvalInfo>())
    {}

    // Checks if the data has been evaluated.
    bool IsEvaluated() const noexcept
    {
        return m_data->m_eval;
    }

    // Returns a mutable reference to the data. Throws an exception if the data is already evaluated.
    TData& MutableData()
    {
        if (IsEvaluated())
        {
            throw std::runtime_error("Data is already evaluated.");
        }
        return m_data->m_data;
    }

    // Marks the data as evaluated. Throws an exception if the data is already evaluated.
    void SetEval()
    {
        if (IsEvaluated())
        {
            throw std::runtime_error("Data is already evaluated.");
        }
        m_data->m_eval = true;
    }

    // Returns a const reference to the data. Throws an exception if the data is not evaluated.
    const TData& Data() const
    {
        if (!IsEvaluated())
        {
            throw std::runtime_error("Data is not evaluated.");
        }
        return m_data->m_data;
    }

    // Returns a const pointer to the underlying data structure (the `DataWithEvalInfo` object).
    const void* DataPtr() const
    {
        return m_data.get();
    }

    // Allocates memory for the data and constructs it with the provided parameters.
    // Throws an exception if the data is already evaluated.
    template <typename...TParams>
    void Allocate(TParams&&... params) const
    {
        if (IsEvaluated())
        {
            throw std::runtime_error("Data is already evaluated.");
        }
        m_data->m_data = TData(std::forward<TParams>(params)...);
    }

private:
    // A shared pointer to the `DataWithEvalInfo` object that holds the data and evaluation flag.
    std::shared_ptr<DataWithEvalInfo> m_data;
};

// The `ConstEvalHandle` class template is used to hold a const version of the data.
// It provides read - only access to the data.
template <typename TData>
class ConstEvalHandle
{
public:
    // Constructor that takes a `TData` object by value and stores it.
    ConstEvalHandle(TData data)
        : m_constData(std::move(data))
    {}

    // Returns a const reference to the stored data.
    const TData& Data() const
    {
        return m_constData;
    }

    // Returns a const pointer to the stored data.
    const void* DataPtr() const
    {
        return &m_constData;
    }

private:
    // The const data object.
    TData m_constData;
};

// Specialization of `ConstEvalHandle` for `EvalHandle<TData>`.
// Allows creating a const handle from an `EvalHandle`.
template <typename TData>
class ConstEvalHandle<EvalHandle<TData>>
{
public:
    // Constructor that takes an `EvalHandle<TData>` object by value and stores it.
    ConstEvalHandle(EvalHandle<TData> data)
        : m_constData(std::move(data))
    {}

    // Returns a const reference to the data through the `EvalHandle`'s `Data` method.
    const TData& Data() const
    {
        return m_constData.Data();
    }

    // Returns a const pointer to the data through the `EvalHandle`'s `DataPtr` method.
    const void* DataPtr() const
    {
        return m_constData.DataPtr();
    }

private:
    // The `EvalHandle` object that holds the data.
    EvalHandle<TData> m_constData;
};

// Function template to create a `ConstEvalHandle` from a const reference to data.
template <typename TData>
auto MakeConstEvalHandle(const TData& data)
{
    return ConstEvalHandle<TData>(data);
}

namespace NSEvalHandle
{
    // `DynamicHandleDataBase` is an abstract base class for dynamic handle data.
    // It provides pure virtual functions for accessing data.
    template <typename TData>
    class DynamicHandleDataBase
    {
    public:
        // Virtual destructor to ensure proper cleanup in derived classes.
        virtual ~DynamicHandleDataBase() = default;
        // Pure virtual function to get a const reference to the data.
        virtual const TData& Data() const = 0;
        // Pure virtual function to get a const pointer to the data.
        virtual const void* DataPtr() const = 0;
    };

    // Forward declaration of `DynamicHandleData` class template.
    template <typename TData>
    class DynamicHandleData;

    // Specialization of `DynamicHandleData` for `ConstEvalHandle<TData>`.
    // It inherits from `DynamicHandleDataBase` and provides concrete implementations for data access.
    template <typename TData>
    class DynamicHandleData<ConstEvalHandle<TData>>
        : public DynamicHandleDataBase<TData>
    {
    public:
        // Constructor that takes a `ConstEvalHandle<TData>` object by value and stores it.
        DynamicHandleData(ConstEvalHandle<TData> data)
            : DynamicHandleDataBase<TData>()
           , m_data(std::move(data)) {}

        // Overrides the `Data` method from the base class to return the data from the `ConstEvalHandle`.
        const TData& Data() const override
        {
            return m_data.Data();
        }

        // Overrides the `DataPtr` method from the base class to return the data pointer from the `ConstEvalHandle`.
        const void* DataPtr() const override
        {
            return m_data.DataPtr();
        }

    private:
        // The `ConstEvalHandle` object that holds the data.
        ConstEvalHandle<TData> m_data;
    };

    // Specialization of `DynamicHandleData` for `ConstEvalHandle<EvalHandle<TData>>`.
    // It inherits from `DynamicHandleDataBase` and provides concrete implementations for data access.
    template <typename TData>
    class DynamicHandleData<ConstEvalHandle<EvalHandle<TData>>>
        : public DynamicHandleDataBase<TData>
    {
    public:
        // Constructor that takes a `ConstEvalHandle<EvalHandle<TData>>` object by value and stores it.
        DynamicHandleData(ConstEvalHandle<EvalHandle<TData>> data)
            : DynamicHandleDataBase<TData>()
           , m_data(std::move(data)) {}

        // Overrides the `Data` method from the base class to return the data from the `ConstEvalHandle`.
        const TData& Data() const override
        {
            return m_data.Data();
        }

        // Overrides the `DataPtr` method from the base class to return the data pointer from the `ConstEvalHandle`.
        const void* DataPtr() const override
        {
            return m_data.DataPtr();
        }

    private:
        // The `ConstEvalHandle<EvalHandle<TData>>` object that holds the data.
        ConstEvalHandle<EvalHandle<TData>> m_data;
    };
}

// The `DynamicConstEvalHandle` class template provides a way to hold different types of const evaluation handles
// in a polymorphic manner.
template <typename TData>
class DynamicConstEvalHandle
{
    // Alias for the base class type.
    using TBaseData = NSEvalHandle::DynamicHandleDataBase<TData>;
public:
    // Constructor that takes a real handle of type `TRealHandle` and stores it in a shared pointer
    // to a `DynamicHandleData` object.
    template <typename TRealHandle>
    DynamicConstEvalHandle(TRealHandle data)
        : m_data(std::make_shared<NSEvalHandle::DynamicHandleData<TRealHandle>>(std::move(data)))
    {
        assert(m_data);
    }

    // Returns a const reference to the data through the polymorphic `Data` method of the base class.
    const TData& Data() const
    {
        return m_data->Data();
    }

    // Returns a const pointer to the data through the polymorphic `DataPtr` method of the base class.
    const void* DataPtr() const
    {
        return m_data->DataPtr();
    }

private:
    // A shared pointer to the base class object that holds the actual data.
    std::shared_ptr<TBaseData> m_data;
};