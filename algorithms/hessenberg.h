#pragma once

#include "householder.h"

namespace matrix_lib::algorithms {
using IndexType = std::size_t;

template <utils::FloatOrComplex T = long double>
struct HessenbergBasis {
    Matrix<T> H;
    Matrix<T> Q;
};

template <utils::FloatOrComplex T = long double>
HessenbergBasis<T> GetHessenbergForm(const MatrixView<T> &matrix) {
    assert(utils::IsSquare(matrix) && "Hessenberg form for square matrices");

    auto Q = Matrix<T>::Identity(matrix.Rows());
    auto H = matrix.Copy();

    for (IndexType col = 0; col < std::min(matrix.Rows(), matrix.Columns()) - 2;
         ++col) {
        auto vec = H.GetSubmatrix(col + 1, H.Rows(), col, col + 1).Copy();
        HouseholderReduction(vec);

        HouseholderLeftReflection(H, vec, col + 1, col);
        HouseholderRightReflection(H, Matrix<T>::Conjugated(vec), col + 1, col);
        HouseholderLeftReflection(Q, vec, col + 1);
    }

    Q.Conjugate();
    H.RoundZeroes();
    return {H, Q};
}

template <utils::FloatOrComplex T = long double>
HessenbergBasis<T> GetHessenbergForm(const Matrix<T> &matrix) {
    return GetHessenbergForm(matrix.View());
}
} // namespace matrix_lib::algorithms
