#pragma once

#include <policies/policy_container.h>

//template <typename...TParams>
//template<template <typename TPolicyCont, typename...> class T, typename...TPolicies>
//using InjectPolicy = T<PolicyContainer<TPolicies...>, TParams...>;

template<template <typename TPolicyCont> class T, typename...TPolicies>
using InjectPolicy = T<PolicyContainer<TPolicies...>>;