#pragma once

#include <complex>
#include <type_traits>

namespace matrix_lib::utils {
template <typename T>
concept FloatingPoint = std::is_floating_point_v<T>;

template <typename T>
struct IsFloatComplexT {
    static constexpr bool value = false;
};

template <FloatingPoint T>
struct IsFloatComplexT<std::complex<T>> {
    static constexpr bool value = true;
};

template <typename T>
constexpr bool IsFloatComplexValue() {
    return IsFloatComplexT<T>::value;
}

template <typename T>
concept FloatOrComplex = FloatingPoint<T> || IsFloatComplexValue<T>();
} // namespace matrix_lib::utils
