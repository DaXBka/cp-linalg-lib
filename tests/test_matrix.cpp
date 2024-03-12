#include <gtest/gtest.h>

#include "helpers.h"

namespace {
template <typename T = long double>
using Complex = std::complex<T>;

template <typename T = long double>
using Matrix = matrix_lib::Matrix<T>;

using matrix_lib::tests::IsEqualMatrices;
using matrix_lib::tests::RandomMatrixGenerator;
using matrix_lib::utils::IsEqualFloating;

TEST(TEST_MATRIX, BasicConstructors) {
    {
        Matrix<float> square(5);
        EXPECT_EQ(square.Rows(), 5);
        EXPECT_EQ(square.Columns(), 5);

        Matrix<double> rect(2, 3);
        EXPECT_EQ(rect.Rows(), 2);
        EXPECT_EQ(rect.Columns(), 3);

        ASSERT_TRUE(IsEqualMatrices(rect, {{0, 0, 0}, {0, 0, 0}}));
    }
    {
        Matrix<long double> matrix = {{1, 2, 3}, {4, 5, 6}};
        ASSERT_TRUE(IsEqualMatrices(matrix, {{1, 2, 3}, {4, 5, 6}}));
    }
    {
        Matrix<Complex<float>> matrix = {{0, 0}, {1, 1}, {2, 2}};
        ASSERT_TRUE(IsEqualMatrices(matrix, {{0, 0}, {1, 1}, {2, 2}}));
    }
}

TEST(TEST_MATRIX, CopySemantics) {
    using Matrix = Matrix<double>;
    Matrix m1(2);

    {
        Matrix m2 = {{1.0, 7.4}, {4.1, 5.6}};
        m1 = m2;
        EXPECT_TRUE(m1 == m2);

        m2(0, 0) = 0.3;
        EXPECT_FALSE(m1 == m2);
    }

    ASSERT_TRUE(IsEqualMatrices(m1, {{1.0, 7.4}, {4.1, 5.6}}));
}

TEST(TEST_MATRIX, MoveSemantics) {
    using Matrix = Matrix<Complex<float>>;
    Matrix m1;

    {
        Matrix m2 = {{{1, -1}, {0, 2}, {-1, 0}, {-2, 1}}};
        m1 = std::move(m2);
    }

    ASSERT_TRUE(IsEqualMatrices(m1, {{{1, -1}, {0, 2}, {-1, 0}, {-2, 1}}}));
}

void CheckArithmeticSum() {
    using Matrix = Matrix<double>;

    Matrix m1 = {{1, 2, 3}, {4, 5, 6}};
    Matrix m2 = {{7, 8, 9}, {10, 11, 12}};
    ASSERT_TRUE(IsEqualMatrices(m1 + m2, {{8, 10, 12}, {14, 16, 18}}));

    m1 += m1;
    m1 += m2;
    ASSERT_TRUE(IsEqualMatrices(m1, {{9, 12, 15}, {18, 21, 24}}));
}

void CheckArithmeticDiff() {
    using Matrix = Matrix<double>;

    Matrix m1 = {{9, 4}, {5, 1}, {12, 9}};
    Matrix m2 = {{-3, 0}, {1, 4}, {6, -12}};
    ASSERT_TRUE(IsEqualMatrices(m1 - m2, {{12, 4}, {4, -3}, {6, 21}}));

    m1 -= m2;
    m1 -= m2;
    ASSERT_TRUE(IsEqualMatrices(m1, {{15, 4}, {3, -7}, {0, 33}}));
}

void CheckArithmeticMulti() {
    using Matrix = Matrix<double>;

    Matrix m1 = {{8, 6, 1}, {8, 5, 1}};
    Matrix m2 = {{1, 2}, {-4, 2}, {0, -3}};

    ASSERT_TRUE(IsEqualMatrices(m1 * m2, {{-16, 25}, {-12, 23}}));
    ASSERT_TRUE(IsEqualMatrices(m2 * m1,
                                {{24, 16, 3}, {-16, -14, -2}, {-24, -15, -3}}));

    auto m3 = m2 * m1;
    auto identity = Matrix::Identity(3);

    EXPECT_TRUE(m3 * identity == m3);
    EXPECT_TRUE(identity * m3 == m3);
}

TEST(TEST_MATRIX, Arithmetic) {
    CheckArithmeticSum();
    CheckArithmeticDiff();
    CheckArithmeticMulti();
}

TEST(TEST_MATRIX, Transpose) {
    using Matrix = Matrix<float>;

    {
        Matrix m1 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        m1.Transpose();
        ASSERT_TRUE(IsEqualMatrices(m1, {{1, 4, 7}, {2, 5, 8}, {3, 6, 9}}));
    }
    {
        Matrix m2 = {{0, 0}, {2, 2}, {4, 4}};
        auto m3 = Matrix::Transposed(m2);

        ASSERT_TRUE(IsEqualMatrices(m2, {{0, 0}, {2, 2}, {4, 4}}));
        ASSERT_TRUE(IsEqualMatrices(m3, {{0, 2, 4}, {0, 2, 4}}));
    }
}

TEST(TEST_MATRIX, Conjugate) {
    using Matrix = Matrix<Complex<double>>;

    {
        Matrix m1 = {{{1, 2}, {7, -3}}, {{0, -1}, {-5, 1}}, {{4, 0}, {2, -2}}};
        m1.Conjugate();
        ASSERT_TRUE(IsEqualMatrices(
            m1, {{{1, -2}, {0, 1}, {4, 0}}, {{7, 3}, {-5, -1}, {2, 2}}}));
    }
    {
        Matrix m2 = {{{1, 0}, {1, 1}}, {{0, -1}, {1, 0}}};
        auto m3 = Matrix::Conjugated(m2);

        ASSERT_TRUE(IsEqualMatrices(m2, {{{1, 0}, {1, 1}}, {{0, -1}, {1, 0}}}));
        ASSERT_TRUE(IsEqualMatrices(m3, {{{1, 0}, {0, 1}}, {{1, -1}, {1, 0}}}));
    }
}

TEST(TEST_MATRIX, ApplyToEach) {
    Matrix<> matrix = Matrix<>::Identity(3);
    matrix.ApplyToEach([](long double &elem) { elem += 10; });

    ASSERT_TRUE(
        IsEqualMatrices(matrix, {{11, 10, 10}, {10, 11, 10}, {10, 10, 11}}));
    matrix.ApplyToEach([&](const long double &elem, size_t i, size_t j) {
        if (i != j) {
            EXPECT_DOUBLE_EQ(elem, 10.0l);
        }
    });

    matrix.ApplyToEach(
        [](long double &elem, size_t i, size_t j) { elem = (i == j); });

    ASSERT_TRUE(matrix == Matrix<>::Identity(3));
}

TEST(TEST_MATRIX, Normalize) {
    Matrix<double> vector = {{3}, {4}};
    ASSERT_TRUE(IsEqualFloating(vector.GetEuclideanNorm(), 5.0));

    vector.Normalize();

    auto scalar_prod = Matrix<double>::Transposed(vector) * vector;
    ASSERT_TRUE(IsEqualFloating(scalar_prod(0, 0), 1.0));

    Matrix<float> other = {{1}, {2}, {3}};
    auto norm_other = Matrix<float>::Normalized(other);

    ASSERT_FALSE(IsEqualFloating(other.GetEuclideanNorm(),
                                 norm_other.GetEuclideanNorm()));
    ASSERT_TRUE(IsEqualFloating(norm_other.GetEuclideanNorm(), 1.0f));
}

TEST(TEST_MATRIX, DiagonalMatrix) {
    std::vector<double> diag = {1, 2, 3, 4, 5};

    auto matrix = Matrix<double>(diag);
    EXPECT_EQ(matrix.Rows(), diag.size());
    EXPECT_EQ(matrix.Columns(), diag.size());

    auto matrix_diag = matrix.GetDiag();
    EXPECT_EQ(diag.size(), matrix_diag.Rows());

    for (size_t i = 0; i < diag.size(); ++i) {
        EXPECT_DOUBLE_EQ(matrix_diag(i, 0), diag[i]);
        EXPECT_DOUBLE_EQ(matrix(i, i), diag[i]);
    }
}

TEST(TEST_MATRIX, Stress) {
    using Type = long double;
    using MatrixGenerator = RandomMatrixGenerator<Type>;
    using Matrix = Matrix<Type>;

    const size_t it_count = 1000u;

    for (int32_t seed = 1; seed < 10; ++seed) {
        MatrixGenerator gen(seed);

        for (size_t it = 0; it < it_count; ++it) {
            try {
                int32_t random_id = (gen.GetRandomValue() + 50) % 4;

                if (random_id == 0) {
                    auto [m1, m2] = gen.GetSquareMatrices();
                    m1 += m2;
                    m2 = m1 + m2;
                } else if (random_id == 1) {
                    auto [m1, m2] = gen.GetSquareMatrices();
                    m1 -= m2;
                    m2 = m1 - m2;
                } else if (random_id == 2) {
                    auto [m1, m2] = gen.GetRectangleMatrices();
                    m1 *= m2;
                    m2 = m1 * Matrix::Transposed(m2);
                } else {
                    auto [m1, m2] = gen.GetRectangleMatrices();
                    m1.Transpose();
                    m2.Conjugate();
                }
            } catch (...) {
                std::cerr << "Stress test FAILED! Seed: " << seed
                          << ", iteration: " << it << std::endl;
                ASSERT_TRUE(false);
                break;
            }
        }
    }
}
} // namespace
