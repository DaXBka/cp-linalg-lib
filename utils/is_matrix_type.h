#pragma once

#include "../types/matrix.h"
#include "../types/matrix_view.h"
#include "is_float_complex.h"

namespace matrix_lib::utils {
template <typename T>
struct IsMatrixT {
    static constexpr bool value = false;
};

template <FloatOrComplex T>
struct IsMatrixT<Matrix<T>> {
    static constexpr bool value = true;
};

template <FloatOrComplex T>
struct IsMatrixT<MatrixView<T>> {
    static constexpr bool value = true;
};

template <typename T>
constexpr bool IsMatrixValue() {
    return IsMatrixT<T>::value;
}

template <typename T>
concept MatrixType = IsMatrixValue<T>();
} // namespace matrix_lib::utils
