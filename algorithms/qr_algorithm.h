#pragma once

#include "../utils/checks.h"
#include "givens.h"
#include "qr_decomposition.h"

namespace matrix_lib::algorithms {
template <utils::FloatOrComplex T = long double>
T GetWilkinsonShift(const MatrixView<T> &matrix) {
    assert(matrix.Rows() == 2 && "Wilkinson shift for 2x2 matrix.");
    assert(matrix.Columns() == 2 && "Wilkinson shift for 2x2 matrix.");
    assert(matrix(0, 1) == matrix(1, 0) &&
           "Wilkinson shift for symmetric matrix.");

    auto d = (matrix(0, 0) - matrix(1, 1)) / 2;
    auto coefficient =
        std::abs(d) + std::sqrt(d * d + matrix(0, 1) * matrix(0, 1));

    return matrix(1, 1) -
           (utils::Sign(d) * matrix(0, 1) * matrix(0, 1)) / coefficient;
}

template <utils::FloatOrComplex T = long double>
T GetWilkinsonShift(const Matrix<T> &matrix) {
    return GetWilkinsonShift(matrix.View());
}

template <utils::FloatOrComplex T = long double>
struct SpectralPair {
    Matrix<T> D;
    Matrix<T> Q;
};

template <utils::FloatOrComplex T = long double>
SpectralPair<T> GetRealSpectralDecomposition(const MatrixView<T> &matrix,
                                         T shift = T{0},
                                         std::size_t it_cnt = 100) {
    assert(utils::IsHermitian(matrix) &&
           "Spectral decomposition for hermitian matrix.");

    auto D = matrix.Copy();
    auto shift_I = Matrix<T>::Identity(D.Rows(), shift);
    auto transform = Matrix<T>::Identity(D.Rows());

    for (std::size_t i = 0; i < it_cnt; ++i) {
        auto [Q, R] = HouseholderQR(D - shift_I);
        D = R * Q + shift_I;
        transform *= Q;

        D.RoundZeroes();
    }

    return {D, transform};
}

template <utils::FloatOrComplex T = long double>
SpectralPair<T> GetRealSpectralDecomposition(const Matrix<T> &matrix,
                                         T shift = T{0},
                                         std::size_t it_cnt = 50) {
    return GetRealSpectralDecomposition(matrix.View(), shift, it_cnt);
}

template <utils::FloatOrComplex T>
struct DiagBasisQR {
    Matrix<T> U;
    Matrix<T> D;
    Matrix<T> VT;
};

template <utils::FloatOrComplex T = long double>
DiagBasisQR<T> BidiagonalAlgorithmQR(const MatrixView<T> &B,
                                     IndexType it_cnt = 30) {
    assert(B.Rows() == 2 && B.Columns() == 2 &&
           "Bigiagonal QR algorithm for 2x2 matrix.");
    assert(utils::IsBidiagonal(B) &&
           "Bigiagonal QR algorithm for bidiagonal matrix.");

    auto S = B.Copy();
    IndexType r = S.Rows();
    IndexType c = S.Columns();
    IndexType size = std::min(r, c);

    auto U = Matrix<T>::Identity(r);
    auto VT = Matrix<T>::Identity(c);

    while (it_cnt--) {
        auto minor = S.GetSubmatrix(r - 2, r, c - 2, c);
        auto BB = Matrix<T>(2);

        BB(0, 0) = minor(0, 0) * minor(0, 0) +
                   ((r >= 3) ? S(r - 3, c - 2) * S(r - 3, c - 2) : T{0});
        BB(1, 0) = minor(0, 0) * minor(0, 1);
        BB(0, 1) = BB(1, 0);
        BB(1, 1) = minor(0, 1) * minor(0, 1) + minor(1, 1) * minor(1, 1);

        auto shift = GetWilkinsonShift(BB);

        for (IndexType i = 0; i < size; ++i) {
            if (i + 1 < S.Columns()) {
                auto f_elem = (i > 0) ? S(i - 1, i) : S(0, 0) * S(0, 0) - shift;
                auto s_elem = (i > 0) ? S(i - 1, i + 1) : S(0, 1) * S(0, 0);

                GivensLeftRotation(VT, i, i + 1, f_elem, s_elem);
                GivensRightRotation(S, i, i + 1, f_elem, s_elem);
            }

            if (i + 1 < S.Rows()) {
                GivensRightRotation(U, i, i + 1, S(i, i), S(i + 1, i));
                GivensLeftRotation(S, i, i + 1, S(i, i), S(i + 1, i));
            }
        }

        S.RoundZeroes();
    }

    return {U, S, VT};
}

template <utils::FloatOrComplex T = long double>
DiagBasisQR<T> BidiagonalAlgorithmQR(const Matrix<T> &B,
                                     IndexType it_cnt = 30) {
    return BidiagonalAlgorithmQR(B.View(), it_cnt);
}
} // namespace matrix_lib::algorithms
