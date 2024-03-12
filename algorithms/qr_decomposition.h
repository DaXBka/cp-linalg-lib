#pragma once

#include "givens.h"
#include "householder.h"

namespace matrix_lib::algorithms {
using IndexType = std::size_t;

template <utils::FloatOrComplex T = long double>
struct PairQR {
    Matrix<T> Q;
    Matrix<T> R;
};

template <utils::FloatOrComplex T>
PairQR<T> HessenbergQR(const MatrixView<T> &matrix);

template <utils::FloatOrComplex T = long double>
PairQR<T> HouseholderQR(const MatrixView<T> &matrix) {
    if (utils::IsHessenberg(matrix)) {
        return HessenbergQR(matrix);
    }

    auto Q = Matrix<T>::Identity(matrix.Rows());
    auto R = matrix.Copy();

    for (IndexType col = 0; col < std::min(matrix.Rows(), matrix.Columns());
         ++col) {
        auto vec = R.GetSubmatrix(col, R.Rows(), col, col + 1).Copy();
        HouseholderReduction(vec);

        HouseholderLeftReflection(R, vec, col, col);
        HouseholderLeftReflection(Q, vec, col);
    }

    Q.Conjugate();
    R.RoundZeroes();
    return {Q, R};
}

template <utils::FloatOrComplex T = long double>
PairQR<T> HouseholderQR(const Matrix<T> &matrix) {
    return HouseholderQR(matrix.View());
}

template <utils::FloatOrComplex T = long double>
PairQR<T> GivensQR(const MatrixView<T> &matrix) {
    if (utils::IsHessenberg(matrix)) {
        return HessenbergQR(matrix);
    }

    auto Q = Matrix<T>::Identity(matrix.Rows());
    auto R = matrix.Copy();

    for (IndexType col = 0; col < std::min(matrix.Rows(), matrix.Columns());
         ++col) {
        for (IndexType row = matrix.Rows() - 2; row + 1 > col; --row) {
            auto first = R(row, col);
            auto second = R(row + 1, col);

            GivensLeftRotation(R, row, row + 1, first, second);
            GivensLeftRotation(Q, row, row + 1, first, second);
        }
    }

    Q.Conjugate();
    R.RoundZeroes();
    return {Q, R};
}

template <utils::FloatOrComplex T = long double>
PairQR<T> GivensQR(const Matrix<T> &matrix) {
    return GivensQR(matrix.View());
}

template <utils::FloatOrComplex T = long double>
PairQR<T> HessenbergQR(const MatrixView<T> &matrix) {
    assert(utils::IsHessenberg(matrix) &&
           "Hessenberg QR for hessenberg form of matrix.");

    auto Q = Matrix<T>::Identity(matrix.Rows());
    auto R = matrix.Copy();

    for (IndexType i = 0; i < std::min(R.Rows() - 1, R.Columns()); ++i) {
        auto first = R(i, i);
        auto second = R(i + 1, i);

        GivensLeftRotation(R, i, i + 1, first, second);
        GivensRightRotation(Q, i, i + 1, first, second);
    }

    R.RoundZeroes();
    return {Q, R};
}

template <utils::FloatOrComplex T = long double>
PairQR<T> HessenbergQR(const Matrix<T> &matrix) {
    return HessenbergQR(matrix.View());
}
} // namespace matrix_lib::algorithms
