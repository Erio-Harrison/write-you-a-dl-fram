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