#pragma once

#include "../types/matrix.h"

namespace matrix_lib::utils {
using IndexType = std::size_t;

template <utils::MatrixType M>
bool IsSquare(const M &matrix) {
    return matrix.Rows() == matrix.Columns();
}

template <utils::MatrixType M>
bool IsUnitary(const M &matrix) {
    if (matrix.Rows() != matrix.Columns()) {
        return false;
    }

    auto prod = matrix * M::Conjugated(matrix);
    return prod == M::Identity(matrix.Rows());
}

template <utils::MatrixType M>
bool IsHermitian(const M &matrix) {
    return matrix == M::Conjugated(matrix);
}

template <utils::MatrixType M>
bool IsNormal(const M &matrix) {
    auto m1 = matrix * M::Conjugated(matrix);
    auto m2 = M::Conjugated(matrix) * matrix;
    return m1 == m2;
}

template <utils::MatrixType M>
bool IsUpperTriangular(const M &matrix) {
    for (IndexType i = 1; i < matrix.Rows(); ++i) {
        for (IndexType j = 0; j < i; ++j) {
            if (!utils::IsZeroFloating(matrix(i, j))) {
                return false;
            }
        }
    }

    return true;
}

template <utils::MatrixType M>
bool IsDiagonal(const M &matrix) {
    for (IndexType i = 0; i < matrix.Rows(); ++i) {
        for (IndexType j = 0; j < matrix.Columns(); ++j) {
            if (i != j && !utils::IsZeroFloating(matrix(i, j))) {
                return false;
            }
        }
    }

    return true;
}

template <utils::MatrixType M>
bool IsBidiagonal(const M &matrix) {
    for (IndexType i = 0; i < matrix.Rows(); ++i) {
        for (IndexType j = 0; j < matrix.Columns(); ++j) {
            if (i != j && i + 1 != j && !utils::IsZeroFloating(matrix(i, j))) {
                return false;
            }
        }
    }

    return true;
}

template <utils::MatrixType M>
bool IsHessenberg(const M &matrix) {
    for (IndexType i = 2; i < matrix.Rows(); ++i) {
        for (IndexType j = 0; j < i - 1; ++j) {
            if (!utils::IsZeroFloating(matrix(i, j))) {
                return false;
            }
        }
    }

    return IsSquare(matrix);
}
} // namespace matrix_lib::utils
