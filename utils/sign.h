#pragma once

#include "is_equal_floating.h"
#include "is_float_complex.h"

namespace matrix_lib::utils {
template <utils::FloatOrComplex T = long double>
T Sign(T value) {
    if constexpr (utils::IsFloatComplexValue<T>()) {
        if (utils::IsZeroFloating(value)) {
            return T{0};
        }
        return value / std::sqrt(std::norm(value));
    } else {
        return (value >= 0) ? T{1} : T{-1};
    }
}
} // namespace matrix_lib::utils
