#ifndef CUSTOSH_UTILITY_H
#define CUSTOSH_UTILITY_H


#include <array>
#include <string>
#include <sstream>
#include <utility>
#include <vector>
#include <cmath>
#include <numbers>

namespace Custosh
{
    extern const std::string ASCIIByBrightness;

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

        [[nodiscard]] unsigned int getNRows() const
        {
            return Rows;
        }

        [[nodiscard]] unsigned int getNCols() const
        {
            return Cols;
        }

        Matrix<T, Rows, Cols> operator*(const T& scalar) const
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

        Matrix<T, Rows, Cols> operator+(const Matrix<T, Rows, Cols>& other) const
        {
            Matrix<T, Rows, Cols> result;

            for (unsigned int i = 0; i < Rows; ++i) {
                for (unsigned int j = 0; j < Cols; ++j) {
                    result(i, j) = m_matrix.at(i).at(j) + other(i, j);
                }
            }

            return result;
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

        [[nodiscard]] Matrix<T, Cols, Rows> transpose() const
        {
            Matrix<T, Cols, Rows> result;

            for (unsigned int i = 0; i < Rows; ++i) {
                for (unsigned int j = 0; j < Cols; ++j) {
                    result(j, i) = m_matrix.at(i).at(j);
                }
            }

            return result;
        }

    protected:
        std::array<std::array<T, Cols>, Rows> m_matrix;

    }; // Matrix

    class PerspectiveMatrix : public Matrix<float, 4, 4>
    {
    public:
        explicit PerspectiveMatrix(float nearPlaneDist, float farPlaneDist) : Matrix<float, 4, 4>()
        {
            (*this)(0, 0) = nearPlaneDist;
            (*this)(1, 1) = nearPlaneDist;
            (*this)(2, 2) = nearPlaneDist + farPlaneDist;
            (*this)(2, 3) = -nearPlaneDist * farPlaneDist;
            (*this)(3, 2) = 1.f;
        }

    }; // PerspectiveMatrix

    template<typename T, unsigned int Size>
    class Vector : public Matrix<T, Size, 1>
    {
    public:
        Vector() : Matrix<T, Size, 1>()
        {
        }

        explicit Vector(Matrix<T, Size, 1> matrix) : Matrix<T, Size, 1>(matrix)
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

        [[nodiscard]] T dot(const Vector<T, Size>& other) const
        {
            return ((*this).transpose() * other)(0, 0);
        }

        [[nodiscard]] T normSq() const
        {
            T normSq = T();

            for (unsigned int i = 0; i < Size; ++i) {
                normSq += (*this)(i) * (*this)(i);
            }

            return normSq;
        }

    }; // Vector

    template<typename T>
    class Vector2 : public Vector<T, 2>
    {
    public:
        Vector2() : Vector<T, 2>()
        {
        }

        explicit Vector2(Vector<T, 2> vector) : Vector<T, 2>(vector)
        {
        }

        explicit Vector2(Matrix<T, 2, 1> matrix) : Vector<T, 2>(matrix)
        {
        }

        Vector2(const std::initializer_list<T>& init) : Vector<T, 2>(init)
        {
        }

        T& x()
        {
            return (*this)(0);
        }

        [[nodiscard]] const T& x() const
        {
            return (*this)(0);
        }

        T& y()
        {
            return (*this)(1);
        }

        [[nodiscard]] const T& y() const
        {
            return (*this)(1);
        }

    }; // Vector2

    template<typename T>
    class Vector4 : public Vector<T, 4>
    {
    public:
        Vector4() : Vector<T, 4>()
        {
        }

        explicit Vector4(Vector<T, 4> vector) : Vector<T, 4>(vector)
        {
        }

        explicit Vector4(Matrix<T, 4, 1> matrix) : Vector<T, 4>(matrix)
        {
        }

        Vector4(const std::initializer_list<T>& init) : Vector<T, 4>(init)
        {
        }

        T& x()
        {
            return (*this)(0);
        }

        [[nodiscard]] const T& x() const
        {
            return (*this)(0);
        }

        T& y()
        {
            return (*this)(1);
        }

        [[nodiscard]] const T& y() const
        {
            return (*this)(1);
        }

        T& z()
        {
            return (*this)(2);
        }

        [[nodiscard]] const T& z() const
        {
            return (*this)(2);
        }

        T& w()
        {
            return (*this)(3);
        }

        [[nodiscard]] const T& w() const
        {
            return (*this)(3);
        }

        [[nodiscard]] Vector4<T> normalizeW() const
        {
            Vector4<T> result;

            result.x() = this->x() / this->w();
            result.y() = this->y() / this->w();
            result.z() = this->z() / this->w();
            result.w() = 1;

            return result;
        }

    }; // Vector4

    template<typename T>
    class Vector3 : public Vector<T, 3>
    {
    public:
        Vector3() : Vector<T, 3>()
        {
        }

        explicit Vector3(Vector<T, 3> vector) : Vector<T, 3>(vector)
        {
        }

        explicit Vector3(Matrix<T, 3, 1> matrix) : Vector<T, 3>(matrix)
        {
        }

        Vector3(const std::initializer_list<T>& init) : Vector<T, 3>(init)
        {
        }

        T& x()
        {
            return (*this)(0);
        }

        [[nodiscard]] const T& x() const
        {
            return (*this)(0);
        }

        T& y()
        {
            return (*this)(1);
        }

        [[nodiscard]] const T& y() const
        {
            return (*this)(1);
        }

        T& z()
        {
            return (*this)(2);
        }

        [[nodiscard]] const T& z() const
        {
            return (*this)(2);
        }

        [[nodiscard]] Vector3<T> cross(const Vector3<T>& other) const
        {
            Vector3<T> result;

            result(0) = (*this)(1) * other(2) - (*this)(2) * other(1);
            result(1) = (*this)(2) * other(0) - (*this)(0) * other(2);
            result(2) = (*this)(0) * other(1) - (*this)(1) * other(0);

            return result;
        }

        [[nodiscard]] Vector4<T> toHomogeneous() const
        {
            return {x(), y(), z(), 1};
        }

    }; // Vector3

    template<typename T>
    class ResizableMatrix
    {
    public:
        ResizableMatrix() : m_rows(0), m_cols(0)
        {
        }

        ResizableMatrix(unsigned int rows, unsigned int cols)
                : m_rows(rows),
                  m_cols(cols),
                  m_matrix(rows, std::vector<T>(cols))
        {
        }

        void resize(unsigned int newRows, unsigned int newCols)
        {
            m_matrix.resize(newRows);
            for (auto& row: m_matrix) {
                row.resize(newCols);
            }
            m_rows = newRows;
            m_cols = newCols;
        }

        T& operator()(unsigned int row, unsigned int col)
        {
            return m_matrix.at(row).at(col);
        }

        const T& operator()(unsigned int row, unsigned int col) const
        {
            return m_matrix.at(row).at(col);
        }

        [[nodiscard]] unsigned int getNRows() const
        {
            return m_rows;
        }

        [[nodiscard]] unsigned int getNCols() const
        {
            return m_cols;
        }

    protected:
        std::vector<std::vector<T>> m_matrix;
        unsigned int m_rows;
        unsigned int m_cols;

    }; // ResizableMatrix

    class BrightnessMap : public ResizableMatrix<float>
    {
    public:
        BrightnessMap() : ResizableMatrix<float>()
        {
        }

        BrightnessMap(unsigned int rows, unsigned int cols) : ResizableMatrix<float>(rows, cols)
        {
        }

        [[nodiscard]] std::string toString() const
        {
            std::ostringstream oss;

            for (unsigned int i = 0; i < m_rows; ++i) {
                for (unsigned int j = 0; j < m_cols; ++j) {
                    oss << brightnessToASCII(m_matrix.at(i).at(j));
                }
                oss << "\n";
            }

            return oss.str();
        }

        friend std::ostream& operator<<(std::ostream& os, const BrightnessMap& bm)
        {
            return os << bm.toString();
        }

    private:
        static char brightnessToASCII(float brightness)
        {
            unsigned int idx = std::ceil(brightness * static_cast<float>(ASCIIByBrightness.size() - 1));
            return ASCIIByBrightness.at(idx);
        }

    }; // BrightnessMap

    template<typename T>
    class Quaternion
    {
    public:
        Quaternion(T real, Vector3<T> imaginaryVec) : m_realPart(real), m_imaginaryVec(std::move(imaginaryVec))
        {
        }

        Quaternion(T real, T i, T j, T k) : m_realPart(real), m_imaginaryVec({i, j, k})
        {
        }

        [[nodiscard]] T getRealPart() const
        {
            return m_realPart;
        }

        [[nodiscard]] Vector3<T> getImaginaryVec() const
        {
            return m_imaginaryVec;
        }

        Quaternion<T> operator*(const T& scalar) const
        {
            return {scalar * m_realPart, Vector3<T>(scalar * m_imaginaryVec)};
        }

        friend Quaternion<T> operator*(const T& scalar, const Quaternion<T>& quaternion)
        {
            return quaternion * scalar;
        }

        Quaternion<T> operator+(const Quaternion& other) const
        {
            return {m_realPart + other.m_realPart, m_imaginaryVec + other.m_imaginaryVec};
        }

        Quaternion<T> operator*(const Quaternion& other) const
        {
            return {m_realPart * other.m_realPart - m_imaginaryVec.dot(other.m_imaginaryVec),
                    Vector3<float>(m_realPart * other.m_imaginaryVec +
                                   other.m_realPart * m_imaginaryVec +
                                   m_imaginaryVec.cross(other.m_imaginaryVec))};
        }

        [[nodiscard]] Quaternion<T> conjunction() const
        {
            return {m_realPart, Vector3<T>(-1 * m_imaginaryVec)};
        }

        [[nodiscard]] T normSq() const
        {
            return m_realPart * m_realPart + m_imaginaryVec.normSq();
        }

    private:
        T m_realPart;
        Vector3<T> m_imaginaryVec;

    }; // Quaternion

    struct triangle3D_t
    {
        Vector3<float> p0;
        Vector3<float> p1;
        Vector3<float> p2;
    };

    struct triangle2D_t
    {
        Vector2<float> p0;
        Vector2<float> p1;
        Vector2<float> p2;
    };

    struct triangleIndices_t
    {
        unsigned int p0;
        unsigned int p1;
        unsigned int p2;
    };

    struct boundingBox_t
    {
        int xMax;
        int xMin;
        int yMax;
        int yMin;

    };

    struct barycentricCoords_t
    {
        float alpha;
        float beta;
        float gamma;
    };

    struct pixel_t
    {
        bool occupied = false;
        Vector3<float> coords;
    };

    struct lightSource_t
    {
        Vector3<float> coords;
        float maxDistanceSq = 1;
    };

    inline float degreesToRadians(float degrees)
    {
        return degrees * (std::numbers::pi_v<float> / 180.f);
    }

} // Custosh


#endif // CUSTOSH_UTILITY_H
