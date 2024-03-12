#pragma once

#include "../types/matrix.h"
#include "../types/matrix_view.h"
#include "../utils/sign.h"

namespace matrix_lib::algorithms {
using IndexType = std::size_t;

template <utils::FloatOrComplex T = long double>
void HouseholderReduction(Matrix<T> &vector) {
    vector(0, 0) += utils::Sign(vector(0, 0)) * vector.GetEuclideanNorm();
    vector.Normalize();
}

template <utils::FloatOrComplex T = long double>
void HouseholderLeftReflection(Matrix<T> &matrix, const Matrix<T> &vec,
                               IndexType row = 0, IndexType c_from = 0,
                               IndexType c_to = -1) {
    if (c_to == -1) {
        c_to = matrix.Columns();
    }

    MatrixView<T> sub =
        matrix.GetSubmatrix(row, row + vec.Rows(), c_from, c_to);
    auto hh_diff = (T{2} * vec) * (Matrix<T>::Conjugated(vec) * sub);
    matrix.AssignSubmatrix(sub.Copy() - hh_diff, row, c_from);
}

template <utils::FloatOrComplex T = long double>
void HouseholderRightReflection(Matrix<T> &matrix, const Matrix<T> &vec,
                                IndexType col = 0, IndexType r_from = 0,
                                IndexType r_to = -1) {
    if (r_to == -1) {
        r_to = matrix.Rows();
    }

    MatrixView<T> sub =
        matrix.GetSubmatrix(r_from, r_to, col, col + vec.Columns());
    auto hh_diff = (sub * Matrix<T>::Conjugated(vec)) * (T{2} * vec);
    matrix.AssignSubmatrix(sub.Copy() - hh_diff, r_from, col);
}
} // namespace matrix_lib::algorithms
