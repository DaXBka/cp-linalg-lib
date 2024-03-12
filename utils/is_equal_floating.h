#pragma once

#include "is_float_complex.h"

#include <cmath>

namespace matrix_lib::utils {
template <FloatOrComplex T>
struct TypeEpsilon {
    static constexpr T kValue = 1e-6;
};

template <>
struct TypeEpsilon<double> {
    static constexpr double kValue = 1e-10;
};

template <>
struct TypeEpsilon<long double> {
    static constexpr long double kValue = 1e-15;
};

template <typename T>
struct TypeEpsilon<std::complex<T>> {
    static constexpr T kValue = TypeEpsilon<T>::kValue;
};

template <FloatOrComplex T>
static constexpr auto Eps = TypeEpsilon<T>::kValue;

template <FloatOrComplex T = long double>
bool IsEqualFloating(T lhs, T rhs) {
    if constexpr (IsFloatComplexValue<T>()) {
        auto is_equal_real = std::abs(lhs.real() - rhs.real()) < Eps<T>;
        auto is_equal_imag = std::abs(lhs.imag() - rhs.imag()) < Eps<T>;
        return is_equal_real && is_equal_imag;
    } else {
        return std::abs(lhs - rhs) < Eps<T>;
    }
}

template <FloatOrComplex T = long double>
bool IsZeroFloating(T lhs) {
    return IsEqualFloating<T>(lhs, T{0});
}
} // namespace matrix_lib::utils
