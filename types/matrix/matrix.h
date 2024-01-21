#pragma once

#include "../../utils/is_float_complex.h"

#include <cassert>
#include <functional>
#include <ostream>
#include <utility>
#include <vector>

namespace matrix_lib {
template <utils::FloatOrComplex T = long double>
class Matrix {
    using IndexType = std::size_t;
    using Data = std::vector<T>;
    using Function = std::function<void(T &)>;
    using FunctionIndexes = std::function<void(T &, IndexType, IndexType)>;

public:
    Matrix() = default;

    explicit Matrix(IndexType sq_size)
        : rows_(sq_size), buffer_(rows_ * rows_, T{0}) {
        assert(Rows() > 0 &&
               "Size of a square matrix must be greater than zero.");
    }

    Matrix(IndexType row_cnt, IndexType col_cnt, T value = T{0})
        : rows_(row_cnt), buffer_(rows_ * col_cnt, value) {
        assert(Rows() > 0 &&
               "Number of matrix rows must be greater than zero.");
        assert(Columns() > 0 &&
               "Number of matrix columns must be greater than zero.");
    }

    explicit Matrix(const Data &diag)
        : rows_(diag.size()), buffer_(rows_ * rows_, T{0}) {
        assert(Rows() > 0 &&
               "List to create a diagonal matrix must not be empty.");

        IndexType idx = 0;
        for (auto value : diag) {
            (*this)(idx, idx) = value;
            ++idx;
        }
    }

    Matrix(std::initializer_list<std::initializer_list<T>> list)
        : rows_(list.size()) {
        assert(Rows() > 0 &&
               "Number of matrix rows must be greater than zero.");
        assert(list.begin()->size() > 0 &&
               "Number of matrix columns must be greater than zero.");

        auto columns = list.begin()->size();
        buffer_.reserve(Rows() * columns);

        for (auto sublist : list) {
            assert(
                sublist.size() == columns &&
                "Size of matrix rows must be equal to the number of columns.");

            for (auto value : sublist) {
                buffer_.push_back(value);
            }
        }
    }

    Matrix(const Matrix &rhs) = default;

    Matrix(Matrix &&rhs) noexcept
        : rows_(std::exchange(rhs.rows_, 0)), buffer_(std::move(rhs.buffer_)) {}

    Matrix &operator=(const Matrix &rhs) = default;

    Matrix &operator=(Matrix &&rhs) noexcept {
        rows_ = std::exchange(rhs.rows_, 0);
        buffer_ = std::move(rhs.buffer_);
        return *this;
    }

    friend Matrix operator+(const Matrix &lhs, const Matrix &rhs) {
        assert(lhs.Rows() == rhs.Rows() &&
               "Number of matrix rows must be equal for addition.");
        assert(rhs.Columns() == rhs.Columns() &&
               "Number of matrix columns must be equal for addition.");

        Matrix res = lhs;
        return res += rhs;
    }

    Matrix &operator+=(const Matrix &rhs) {
        assert(Rows() == rhs.Rows() &&
               "Number of matrix rows must be equal for addition.");
        assert(Columns() == rhs.Columns() &&
               "Number of matrix columns must be equal for addition.");

        for (IndexType i = 0; i < buffer_.size(); ++i) {
            buffer_[i] += rhs.buffer_[i];
        }

        return *this;
    }

    friend Matrix operator-(const Matrix &lhs, const Matrix &rhs) {
        assert(lhs.Rows() == rhs.Rows() &&
               "Number of matrix rows must be equal for subtraction.");
        assert(lhs.Columns() == rhs.Columns() &&
               "Number of matrix columns must be equal for subtraction.");

        Matrix res = lhs;
        return res -= rhs;
    }

    Matrix &operator-=(const Matrix &rhs) {
        assert(Rows() == rhs.Rows() &&
               "Number of matrix rows must be equal for subtraction.");
        assert(Columns() == rhs.Columns() &&
               "Number of matrix columns must be equal for subtraction.");

        for (IndexType i = 0; i < buffer_.size(); ++i) {
            buffer_[i] -= rhs.buffer_[i];
        }

        return *this;
    }

    friend Matrix operator*(const Matrix &lhs, const Matrix &rhs) {
        assert(lhs.Columns() == rhs.Rows() &&
               "Matrix dimension mismatch for multiplication.");
        Matrix result(lhs.Rows(), rhs.Columns());

        for (IndexType i = 0; i < lhs.Rows(); ++i) {
            for (IndexType j = 0; j < rhs.Columns(); ++j) {
                T sum = 0;
                for (IndexType k = 0; k < lhs.Columns(); ++k) {
                    sum += lhs(i, k) * rhs(k, j);
                }

                result(i, j) = sum;
            }
        }

        return result;
    }

    Matrix &operator*=(const Matrix &rhs) { return *this = *this * rhs; }

    friend Matrix operator*(const Matrix &lhs, T scalar) {
        Matrix res = lhs;
        return res *= scalar;
    }

    Matrix &operator*=(T scalar) {
        ApplyToEach([scalar](T &value) { value *= scalar; });
        return *this;
    }

    friend bool operator==(const Matrix &lhs, const Matrix &rhs) {
        return lhs.buffer_ == rhs.buffer_;
    }

    friend bool operator!=(const Matrix &lhs, const Matrix &rhs) {
        return !(lhs == rhs);
    }

    friend bool operator<=>(const Matrix &lhs, const Matrix &rhs) = delete;

    T &operator()(IndexType row_idx, IndexType col_idx) {
        assert(Columns() * row_idx + col_idx < buffer_.size() &&
               "Requested indexes are outside the matrix boundaries.");
        return buffer_[Columns() * row_idx + col_idx];
    }

    T operator()(IndexType row_idx, IndexType col_idx) const {
        assert(Columns() * row_idx + col_idx < buffer_.size() &&
               "Requested indexes are outside the matrix boundaries.");
        return buffer_[Columns() * row_idx + col_idx];
    }

    [[nodiscard]] IndexType Rows() const { return rows_; }

    [[nodiscard]] IndexType Columns() const {
        return (rows_ == 0) ? 0 : buffer_.size() / rows_;
    }

    void ApplyToEach(Function func) {
        for (IndexType i = 0; i < buffer_.size(); ++i) {
            func(buffer_[i]);
        }
    }

    void ApplyToEach(FunctionIndexes func) {
        for (IndexType i = 0; i < buffer_.size(); ++i) {
            func(buffer_[i], i / Rows(), i % Rows());
        }
    }

    Matrix GetDiag(bool transpose = false) const {
        auto size = std::min(Rows(), Columns());

        Matrix res(size, 1);
        for (IndexType i = 0; i < size; ++i) {
            res(i, 0) = (*this)(i, i);
        }

        if (transpose) {
            res.Transpose();
        }

        return res;
    }

    Matrix GetRow(IndexType index) const {
        assert(index < Rows() && "Index must be less than the number of matrix rows.");

        Matrix res(1, Columns());
        for (IndexType i = 0; i < Columns(); ++i) {
            res(0, i) = (*this)(index, i);
        }

        return res;
    }

    Matrix GetColumn(IndexType index) const {
        assert(index < Columns() && "Index must be less than the number of matrix columns.");

        Matrix res(Rows(), 1);
        for (IndexType i = 0; i < Rows(); ++i) {
            res(i, 0) = (*this)(i, index);
        }

        return res;
    }

    void Transpose() {
        std::vector<bool> visited(buffer_.size(), false);
        IndexType last_idx = buffer_.size() - 1;

        for (IndexType i = 1; i < buffer_.size(); ++i) {
            if (visited[i]) {
                continue;
            }

            auto swap_idx = i;
            do {
                swap_idx = (swap_idx == last_idx)
                               ? last_idx
                               : (Rows() * swap_idx) % last_idx;
                std::swap(buffer_[swap_idx], buffer_[i]);
                visited[swap_idx] = true;
            } while (swap_idx != i);
        }

        rows_ = Columns();
    }

    void Conjugate() {
        Transpose();

        if constexpr (utils::IsFloatComplexValue<T>()) {
            ApplyToEach([](T &val) { val = std::conj(val); });
        }
    }

    static Matrix Identity(IndexType size, T default_value = T{1}) {
        return Matrix(Data(size, default_value));
    }

    static Matrix Transposed(const Matrix &rhs) {
        Matrix res = rhs;
        res.Transpose();
        return res;
    }

    static Matrix Conjugated(const Matrix &rhs) {
        Matrix res = rhs;
        res.Conjugate();
        return res;
    }

    friend std::ostream &operator<<(std::ostream &ostream,
                                    const Matrix &matrix) {
        ostream << '[';
        for (std::size_t i = 0; i < matrix.Rows(); ++i) {
            ostream << '[';
            for (std::size_t j = 0; j < matrix.Columns(); ++j) {
                ostream << matrix(i, j);
                if (j + 1 < matrix.Columns()) {
                    ostream << ' ';
                }
            }

            ostream << ']';
            if (i + 1 < matrix.Rows()) {
                ostream << '\n';
            }
        }
        ostream << ']';
        return ostream;
    }

private:
    IndexType rows_ = 0;
    Data buffer_;
};
} // namespace matrix_lib
