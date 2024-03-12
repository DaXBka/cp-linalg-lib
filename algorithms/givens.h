#pragma once

#include "../types/matrix.h"

namespace matrix_lib::algorithms {
using IndexType = std::size_t;

template <utils::FloatOrComplex T = long double>
struct GivensPair {
    T cos = T{0};
    T sin = T{0};
};

template <utils::FloatOrComplex T = long double>
GivensPair<T> GetGivensCoefficients(T first_elem, T second_elem) {
    auto abs_first = std::abs(first_elem);
    auto abs_second = std::abs(second_elem);
    auto sqrt_abs = std::sqrt(abs_first * abs_first + abs_second * abs_second);

    if (utils::IsZeroFloating(sqrt_abs)) {
        return {T{1}, T{0}};
    }

    return {first_elem / sqrt_abs, -second_elem / sqrt_abs};
}

template <utils::FloatOrComplex T = long double>
Matrix<T> GetGivensMatrix(IndexType size, IndexType from, IndexType to,
                          T first_elem, T second_elem) {
    auto res = Matrix<T>::Identity(size);
    auto [cos, sin] = GetGivensCoefficients(first_elem, second_elem);

    res(from, from) = cos;
    res(from, to) = -sin;

    if constexpr (utils::IsFloatComplexValue<T>()) {
        res(to, from) = std::conj(sin);
        res(to, to) = std::conj(cos);
    } else {
        res(to, from) = sin;
        res(to, to) = cos;
    }

    return res;
}

template <utils::FloatOrComplex T = long double>
void GivensLeftRotation(Matrix<T> &matrix, IndexType from, IndexType to,
                        T first, T second) {
    auto [cos, sin] = GetGivensCoefficients(first, second);

    for (IndexType i = 0; i < matrix.Columns(); ++i) {
        auto cp_from = matrix(from, i);
        auto cp_to = matrix(to, i);

        matrix(from, i) = cos * cp_from - sin * cp_to;
        matrix(to, i) = cos * cp_to + sin * cp_from;
    }
}

template <utils::FloatOrComplex T = long double>
void GivensRightRotation(Matrix<T> &matrix, IndexType from, IndexType to,
                         T first, T second) {
    auto [cos, sin] = GetGivensCoefficients(first, second);

    for (IndexType i = 0; i < matrix.Rows(); ++i) {
        auto cp_from = matrix(i, from);
        auto cp_to = matrix(i, to);

        matrix(i, from) = cos * cp_from - sin * cp_to;
        matrix(i, to) = cos * cp_to + sin * cp_from;
    }
}
} // namespace matrix_lib::algorithms
