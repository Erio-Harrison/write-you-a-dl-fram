#pragma once

#include <data/facilities/traits.h>
#include <data/facilities/tags.h>
#include <data/facilities/lower_access.h>
#include <evaluate/facilities/eval_handle.h>
#include <data/facilities/continuous_memory.h>

template <typename TElem, typename TDevice = DeviceTags::CPU>
class Scalar
{
    static_assert(std::is_same<RemConstRef<TElem>, TElem>::value);
public:
    using ElementType = TElem;
    using DeviceType = TDevice;
    
public:
    Scalar(ElementType elem = ElementType())
        : m_elem(elem) {}
     
    auto& Value()
    {
        return m_elem;
    }
   
    auto Value() const
    {
        return m_elem;
    }
    
    bool operator== (const Scalar& val) const
    {
        return m_elem == val.m_elem;
    }

    template <typename TOtherType>
    bool operator== (const TOtherType&) const
    {
        return false;
    }

    template <typename TData>
    bool operator!= (const TData& val) const
    {
        return !(operator==(val));
    }
    
    auto EvalRegister() const
    {
        return MakeConstEvalHandle(*this);
    }
    
private:
    ElementType m_elem;
};

// template <typename TElem, typename TDevice = DeviceTags::GPU>
// class Scalar
// {
//     static_assert(std::is_same<RemConstRef<TElem>, TElem>::value);

// public:
//     using ElementType = TElem;
//     using DeviceType = TDevice;

// public:
//     Scalar(ElementType elem = ElementType())
//         : m_elem(elem)
//     {
//         m_gpuElem = Allocator<DeviceTags::GPU>::Allocate<ElementType>(1);
//         cudaMemcpy(m_gpuElem, &m_elem, sizeof(ElementType), cudaMemcpyHostToDevice);
//     }

//     ~Scalar()
//     {
//         // No need to manually free the memory since shared_ptr handles it
//     }

//     auto& Value()
//     {
//         cudaMemcpy(&m_elem, m_gpuElem, sizeof(ElementType), cudaMemcpyDeviceToHost);
//         return m_elem;
//     }

//     auto Value() const
//     {
//         ElementType temp;
//         cudaMemcpy(&temp, m_gpuElem, sizeof(ElementType), cudaMemcpyDeviceToHost);
//         return temp;
//     }

//     bool operator==(const Scalar& val) const
//     {
//         ElementType otherElem;
//         cudaMemcpy(&otherElem, val.m_gpuElem, sizeof(ElementType), cudaMemcpyDeviceToHost);
//         return m_elem == otherElem;
//     }

//     template <typename TOtherType>
//     bool operator==(const TOtherType&) const
//     {
//         return false;
//     }

//     template <typename TData>
//     bool operator!=(const TData& val) const
//     {
//         return !(operator==(val));
//     }

//     auto EvalRegister() const
//     {
//         return MakeConstEvalHandle(*this);
//     }

// private:
//     ElementType m_elem;
//     ElementType* m_gpuElem = nullptr;
// };


template <typename TElem>
class Batch<TElem, DeviceTags::CPU, CategoryTags::Scalar>
{
public:
    using ElementType = TElem;
    using DeviceType = DeviceTags::CPU;
    
    friend LowerAccessImpl<Batch<TElem, DeviceTags::CPU, CategoryTags::Scalar>>;
    
public:
    Batch(size_t length = 0)
        : m_mem(length)
        , m_len(length) {}
     
    size_t BatchNum() const { return m_len; }

    bool AvailableForWrite() const { return m_mem.UseCount() == 1; }

    void SetValue(size_t p_id, ElementType val)
    {
        assert(AvailableForWrite());
        assert(p_id < m_len);
        (m_mem.RawMemory())[p_id] = val;
    }
    
    const auto operator[](size_t p_id) const
    {
        assert(p_id < m_len);
        return (m_mem.RawMemory())[p_id];
    }
   
    bool operator== (const Batch& val) const
    {
        return (m_mem == val.m_mem) && (m_len == val.m_len);
    }

    template <typename TOtherType>
    bool operator== (const TOtherType&) const
    {
        return false;
    }

    template <typename TData>
    bool operator!= (const TData& val) const
    {
        return !(operator==(val));
    }
    
    auto EvalRegister() const
    {
        return MakeConstEvalHandle(*this);
    }
    
private:
    ContinuousMemory<ElementType, DeviceType> m_mem;
    size_t m_len;
};

template<typename TElem>
struct LowerAccessImpl<Batch<TElem, DeviceTags::CPU, CategoryTags::Scalar>>
{
    LowerAccessImpl(Batch<TElem, DeviceTags::CPU, CategoryTags::Scalar> p)
        : m_data(std::move(p))
    {}

    auto MutableRawMemory()
    {
        return m_data.m_mem.RawMemory();
    }

    const auto RawMemory() const
    {
        return m_data.m_mem.RawMemory();
    }

private:
    Batch<TElem, DeviceTags::CPU, CategoryTags::Scalar> m_data;
};