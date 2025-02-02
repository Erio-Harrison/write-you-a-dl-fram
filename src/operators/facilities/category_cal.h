#pragma once

#include <data/facilities/traits.h>
#include <tuple>

template <typename TOpTag, typename THeadCate, typename...TRemainCate>
struct OperCategory_;

template <typename TCate, typename...TRemain>
struct SameCate_ {
    constexpr static bool value = true;
};

template <typename TCate, typename TCur, typename...TRemain>
struct SameCate_<TCate, TCur, TRemain...> {
    constexpr static bool tmp = std::is_same<TCate, TCur>::value;
    template <bool A, bool B>
    struct AndValue {
        constexpr static bool value = A && B;
    };
    constexpr static bool value = AndValue<tmp, SameCate_<TCate, TRemain...>::value>::value;
};

template <typename TCateCont, typename...TData>
struct Data2Cate_ {
    using type = TCateCont;
};

template <typename...TProcessed, typename TCur, typename...TRemain>
struct Data2Cate_<std::tuple<TProcessed...>, TCur, TRemain...> {
    using tmp1 = DataCategory<TCur>;
    using tmp2 = std::tuple<TProcessed..., tmp1>;
    using type = typename Data2Cate_<tmp2, TRemain...>::type;
};

template <typename THead, typename...TRemain>
using Data2Cate = typename Data2Cate_<std::tuple<>, THead, TRemain...>::type;

template <typename TOpTag, typename TCateContainer>
struct CateInduce_;

template <typename TOpTag, typename...TCates>
struct CateInduce_<TOpTag, std::tuple<TCates...>> {
    using type = typename OperCategory_<TOpTag, TCates...>::type;
};

template <typename TOpTag, typename THeadCate, typename...TRemainCate>
struct OperCategory_ {
    static_assert(SameCate_<THeadCate, TRemainCate...>::value,
                  "Data category mismatch.");
    using type = THeadCate;
};

template <typename TOpTag, typename THead, typename...TRemain>
using OperCateCal = typename CateInduce_<TOpTag, Data2Cate<THead, TRemain...>>::type;