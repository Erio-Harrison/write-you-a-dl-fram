#pragma once

#include <unordered_map>
#include <data/facilities/tags.h>
#include <vector>

template <typename TDevice>
class BaseEvalUnit
{
public:
    using DeviceType = TDevice;
    virtual ~BaseEvalUnit() = default;

    virtual void Eval() = 0;
};