#pragma once

#include "householder.h"

namespace matrix_lib::algorithms {
using IndexType = std::size_t;

template <utils::FloatOrComplex T = long double>
struct BidiagonalBasis {
    Matrix<T> U;
    Matrix<T> B;
    Matrix<T> VT;
};

template <utils::FloatOrComplex T = long double>
BidiagonalBasis<T> Bidiagonalize(const MatrixView<T> &matrix) {
    Matrix<T> B = matrix.Copy();
    Matrix<T> U = Matrix<T>::Identity(B.Rows());
    Matrix<T> V = Matrix<T>::Identity(B.Columns());

    for (IndexType col = 0; col < std::min(B.Rows(), B.Columns()); ++col) {
        auto col_reduction = B.GetSubmatrix(col, B.Rows(), col, col + 1).Copy();
        HouseholderReduction(col_reduction);

        HouseholderLeftReflection(B, col_reduction, col, col);
        HouseholderLeftReflection(U, col_reduction, col);

        if (col + 2 < B.Columns()) {
            auto row_reduction =
                B.GetSubmatrix(col, col + 1, col + 1, B.Columns()).Copy();
            HouseholderReduction(row_reduction);

            HouseholderRightReflection(B, row_reduction, col + 1, col);
            HouseholderRightReflection(V, row_reduction, col + 1);
        }
    }

    U.Conjugate();
    V.Conjugate();
    B.RoundZeroes();
    return {U, B, V};
}

template <utils::FloatOrComplex T = long double>
BidiagonalBasis<T> Bidiagonalize(const Matrix<T> &matrix) {
    return Bidiagonalize(matrix.View());
}
} // namespace matrix_lib::algorithms
