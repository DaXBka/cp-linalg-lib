#pragma once

#include "../utils/is_equal_floating.h"
#include "../utils/is_float_complex.h"

#include <cassert>
#include <functional>
#include <utility>
#include <vector>

namespace matrix_lib {
template <utils::FloatOrComplex T>
class Matrix;

template <utils::FloatOrComplex T = long double>
class MatrixView {
    using IndexType = std::size_t;
    using ConstFunction = std::function<void(const T &)>;
    using ConstFunctionIndexes =
        std::function<void(const T &, IndexType, IndexType)>;

    struct IndexPair {
        IndexType from = 0;
        IndexType to = 0;
    };

public:
    explicit MatrixView(const Matrix<T> &matrix, IndexType r_from = 0,
                        IndexType r_to = 0, IndexType c_from = 0,
                        IndexType c_to = 0)
        : matrix_(matrix), row_(r_from, r_to), column_(c_from, c_to) {

        if (r_from > r_to || r_from > matrix_.Rows()) {
            row_.from = 0;
        }

        if (r_to == 0 || r_to > matrix_.Rows()) {
            row_.to = matrix.Rows();
        }

        if (c_from > c_to || c_from > matrix_.Columns()) {
            column_.from = 0;
        }

        if (c_to == 0 || c_to > matrix_.Columns()) {
            column_.to = matrix.Columns();
        }
    }

    MatrixView(const MatrixView &rhs) = default;

    MatrixView(MatrixView &&rhs) noexcept
        : matrix_(std::move(rhs.matrix_)),
          row_(std::exchange(rhs.row_, {0, 0})),
          column_(std::exchange(rhs.column_, {0, 0})){};

    MatrixView &operator=(const MatrixView &lhs) = default;

    MatrixView &operator=(MatrixView &&rhs) noexcept {
        matrix_ = std::move(rhs.matrix_);
        row_ = std::exchange(rhs.row_, {0, 0});
        column_ = std::exchange(rhs.column_, {0, 0});
        return *this;
    }

    T operator()(IndexType row_idx, IndexType col_idx) const {
        return matrix_(row_.from + row_idx, column_.from + col_idx);
    }

    [[nodiscard]] IndexType Rows() const { return row_.to - row_.from; }

    [[nodiscard]] IndexType Columns() const {
        return column_.to - column_.from;
    }

    void ApplyToEach(ConstFunction func) const {
        for (IndexType i = row_.from; i < row_.to; ++i) {
            for (IndexType j = column_.from; j < column_.to; ++j) {
                func(matrix_(i, j));
            }
        }
    }

    void ApplyToEach(ConstFunctionIndexes func) const {
        for (IndexType i = row_.from; i < row_.to; ++i) {
            for (IndexType j = column_.from; j < column_.to; ++j) {
                func(matrix_(i, j), i - row_.from, j - column_.from);
            }
        }
    }

    T GetEuclideanNorm() const {
        assert(Rows() == 1 ||
               Columns() == 1 && "Euclidean norm only for vectors.");

        T sq_sum = T{0};

        if constexpr (utils::IsFloatComplexValue<T>()) {
            ApplyToEach([&](const T &value) { sq_sum += std::norm(value); });
        } else {
            ApplyToEach([&](const T &value) { sq_sum += (value * value); });
        }

        return std::sqrt(sq_sum);
    }

    Matrix<T> GetDiag(bool to_row = false) const {
        auto size = std::min(Rows(), Columns());

        Matrix<T> res(size, 1);
        for (IndexType i = 0; i < size; ++i) {
            res(i, 0) = (*this)(i, i);
        }

        if (to_row) {
            res.Transpose();
        }

        return res;
    }

    MatrixView GetRow(IndexType index) const {
        assert(index < Rows() &&
               "Index must be less than the number of matrix rows.");

        return MatrixView(matrix_, row_.from + index, row_.from + index + 1,
                          column_.from, column_.to);
    }

    MatrixView GetColumn(IndexType index) const {
        assert(index < Columns() &&
               "Index must be less than the number of matrix columns.");

        return MatrixView(matrix_, row_.from, row_.from + row_.to,
                          column_.from + index, column_.from + index + 1);
    }

    Matrix<T> Copy() const {
        Matrix<T> res(Rows(), Columns());
        res.ApplyToEach(
            [&](T &val, IndexType i, IndexType j) { val = (*this)(i, j); });
        return res;
    }

    static Matrix<T> Transposed(const MatrixView<T> &rhs) {
        return Matrix<T>::Transposed(rhs.Copy());
    }

    static Matrix<T> Conjugated(const MatrixView<T> &rhs) {
        return Matrix<T>::Conjugated(rhs.Copy());
    }

    static Matrix<T> Normalized(const MatrixView<T> &rhs) {
        return Matrix<T>::Normalized(rhs.Copy());
    }

    static Matrix<T> Identity(IndexType size, T default_value = T{1}) {
        return Matrix<T>::Identity(size, default_value);
    }

    friend std::ostream &operator<<(std::ostream &ostream,
                                    const MatrixView &matrix) {
        ostream << '{';
        for (std::size_t i = 0; i < matrix.Rows(); ++i) {
            ostream << '{';
            for (std::size_t j = 0; j < matrix.Columns(); ++j) {
                ostream << matrix(i, j);
                if (j + 1 < matrix.Columns()) {
                    ostream << ' ';
                }
            }

            ostream << '}';
            if (i + 1 < matrix.Rows()) {
                ostream << '\n';
            }
        }
        ostream << '}';
        return ostream;
    }

private:
    const Matrix<T> &matrix_;
    IndexPair row_;
    IndexPair column_;
};
} // namespace matrix_lib
