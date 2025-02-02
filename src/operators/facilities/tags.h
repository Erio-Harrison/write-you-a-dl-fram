#pragma once

template <typename...TCases>
struct OperSeqContainer;

template <typename TOpTag>
struct OperSeq_;

struct UnaryOpTags
{
    struct Abs;
    struct Sigmoid;
    struct Sign;
    struct Tanh;
    struct Transpose;
    struct Collapse;
    struct VecSoftmax;
};

struct BinaryOpTags
{
    struct Add;
    struct Substract;
    struct ElementMul;
    struct Divide;
    struct Dot;
    struct NegativeLogLikelihood;
    struct SigmoidDerivative;
    struct TanhDerivative;
    struct VecSoftmaxDerivative;
};

struct TernaryOpTags
{
    struct Interpolate;
    struct NegativeLogLikelihoodDerivative;
};

template <typename TOpTag, typename TOp1, typename...TOperands>
struct OperElementType_
{
    using type = typename TOp1::ElementType;
};

template <typename TOpTag, typename TOp1, typename...TOperands>
struct OperDeviceType_
{
    using type = typename TOp1::DeviceType;
};