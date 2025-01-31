#pragma once

#include <data/facilities/traits.h>
#include <data/facilities/tags.h>
#include <evaluate/facilities/eval_handle.h>

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