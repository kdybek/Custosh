#ifndef CUSTOSH_MATRIX_H
#define CUSTOSH_MATRIX_H


#include <array>
#include <string>
#include <sstream>

namespace Custosh
{
    template<typename T, unsigned int Rows, unsigned int Cols>
    class Matrix
    {
    public:
        Matrix()
        {
            for (unsigned int i = 0; i < Rows; ++i) {
                for (unsigned int j = 0; j < Cols; ++j) {
                    m_matrix.at(i).at(j) = T();
                }
            }
        }

        Matrix(const std::initializer_list<std::initializer_list<T>>& init)
        {
            unsigned int i = 0;
            for (const auto& row: init) {
                unsigned int j = 0;
                for (const auto& elem: row) {
                    m_matrix.at(i).at(j) = elem;
                    ++j;
                }
                ++i;
            }
        }

        T& operator()(unsigned int row, unsigned int col)
        {
            return m_matrix.at(row).at(col);
        }

        const T& operator()(unsigned int row, unsigned int col) const
        {
            return m_matrix.at(row).at(col);
        }

        virtual Matrix<T, Rows, Cols> operator*(const T& scalar) const
        {
            Matrix<T, Rows, Cols> result;

            for (unsigned int i = 0; i < Rows; ++i) {
                for (unsigned int j = 0; j < Cols; ++j) {
                    result(i, j) = scalar * m_matrix.at(i).at(j);
                }
            }

            return result;
        }

        friend Matrix<T, Rows, Cols> operator*(const T& scalar, const Matrix& matrix)
        {
            return matrix * scalar;
        }

        template<unsigned int OtherCols>
        Matrix<T, Rows, OtherCols> operator*(const Matrix<T, Cols, OtherCols>& other) const
        {
            Matrix<T, Rows, OtherCols> result;

            for (unsigned int i = 0; i < Rows; ++i) {
                for (unsigned int j = 0; j < OtherCols; ++j) {
                    result(i, j) = T();
                    for (unsigned int k = 0; k < Cols; ++k) {
                        result(i, j) += m_matrix.at(i).at(k) * other(k, j);
                    }
                }
            }

            return result;
        }

        Matrix<T, Cols, Rows> transpose() const
        {
            Matrix<T, Cols, Rows> result;

            for (unsigned int i = 0; i < Rows; ++i) {
                for (unsigned int j = 0; j < Cols; ++j) {
                    result(j, i) = m_matrix.at(i).at(j);
                }
            }

            return result;
        }

        [[nodiscard]] std::string toString() const
        {
            std::ostringstream oss;

            for (unsigned int i = 0; i < Rows; ++i) {
                for (unsigned int j = 0; j < Cols; ++j) {
                    oss << m_matrix.at(i).at(j) << " ";
                }
                oss << "\n";
            }

            return oss.str();
        }

        friend std::ostream& operator<<(std::ostream& os, Matrix<T, Rows, Cols> matrix)
        {
            return os << matrix.toString();
        }

    protected:
        std::array<std::array<T, Cols>, Rows> m_matrix;

    }; // Matrix

    template<typename T, unsigned int Size>
    class Vector : public Matrix<T, Size, 1>
    {
    public:
        Vector() : Matrix<T, Size, 1>()
        {
        }

        Vector(const std::initializer_list<T>& init)
        {
            unsigned int i = 0;
            for (const auto& elem: init) {
                this->m_matrix.at(i).at(0) = elem;
                ++i;
            }
        }

        T& operator()(unsigned int index)
        {
            return this->m_matrix.at(index).at(0);
        }

        const T& operator()(unsigned int index) const
        {
            return this->m_matrix.at(index).at(0);
        }

        T dot(const Vector<T, Size>& other) const
        {
            return ((*this).transpose() * other)(0, 0);
        }

    }; // Vector

    template<typename T>
    class Vector3 : public Vector<T, 3>
    {
    public:
        Vector3() : Vector<T, 3>()
        {
        }

        Vector3(const std::initializer_list<T>& init) : Vector<T, 3>(init)
        {
        }

        Vector3<T> cross(const Vector3<T>& other) const
        {
            Vector3<T> result;

            result(0) = (*this)(1) * other(2) - (*this)(2) * other(1);
            result(1) = (*this)(2) * other(0) - (*this)(0) * other(2);
            result(2) = (*this)(0) * other(1) - (*this)(1) * other(0);

            return result;
        }

    }; // Vector3

} // Custosh


#endif // CUSTOSH_MATRIX_H
