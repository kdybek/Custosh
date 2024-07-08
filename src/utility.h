#ifndef CUSTOSH_UTILITY_H
#define CUSTOSH_UTILITY_H


#include <string>
#include <utility>
#include <vector>
#include <cmath>
#include <numbers>

#include "custosh_except.h"

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define CUSTOSH_IF_CUDACC(x) x
#else
#define CUSTOSH_IF_CUDACC(x)
#endif

#ifdef __CUDA_ARCH__
#define CUSTOSH_HOST_DEV_ERR(message) \
    do { \
        printf(message); \
        asm("trap;"); \
    } while(0)
#else
#define CUSTOSH_HOST_DEV_ERR(message) \
    do { \
        std::string errMsg = "Error at "; \
        errMsg += __FILE__; \
        errMsg += ":"; \
        errMsg += std::to_string(__LINE__); \
        errMsg += " - "; \
        errMsg += message; \
        throw CustoshException(message); \
    } while(0)
#endif

#define CUSTOSH_INIT_LIST_ERR_MSG "incorrect initializer list"
#define CUSTOSH_IDX_ERR_MSG "index out of bounds"

#define CUSTOSH_HOST_DEV_AUX_FUNC CUSTOSH_IF_CUDACC(__host__ __device__) inline constexpr
#define CUSTOSH_HOST_DEV_MEMBER CUSTOSH_IF_CUDACC(__host__ __device__) constexpr
#define CUSTOSH_HOST_DEV_GETTER CUSTOSH_IF_CUDACC(__host__ __device__) inline constexpr

namespace custosh
{
    /* Functions */
    [[nodiscard]] CUSTOSH_HOST_DEV_AUX_FUNC float degreesToRadians(float degrees)
    {
        return degrees * (std::numbers::pi_v<float> / 180.f);
    }

    /* Classes */
    template<typename T, unsigned int Rows, unsigned int Cols>
    class Matrix
    {
    public:
        CUSTOSH_HOST_DEV_MEMBER Matrix()
        {
            for (unsigned int i = 0; i < Rows; ++i) {
                for (unsigned int j = 0; j < Cols; ++j) {
                    m_matrix[i][j] = T();
                }
            }
        }

        CUSTOSH_HOST_DEV_MEMBER Matrix(const std::initializer_list<std::initializer_list<T>>& init)
        {
            if (init.size() != Rows) { CUSTOSH_HOST_DEV_ERR(CUSTOSH_INIT_LIST_ERR_MSG); }

            unsigned int i = 0;
            for (const auto& row: init) {
                if (row.size() != Cols) { CUSTOSH_HOST_DEV_ERR(CUSTOSH_INIT_LIST_ERR_MSG); }

                unsigned int j = 0;
                for (const auto& elem: row) {
                    m_matrix[i][j] = elem;
                    ++j;
                }
                ++i;
            }
        }

        [[nodiscard]] CUSTOSH_HOST_DEV_MEMBER T& operator()(unsigned int row, unsigned int col)
        {
            if (row >= Rows || col >= Cols) { CUSTOSH_HOST_DEV_ERR(CUSTOSH_IDX_ERR_MSG); }

            return m_matrix[row][col];
        }

        [[nodiscard]] CUSTOSH_HOST_DEV_MEMBER const T& operator()(unsigned int row, unsigned int col) const
        {
            if (row >= Rows || col >= Cols) { CUSTOSH_HOST_DEV_ERR(CUSTOSH_IDX_ERR_MSG); }

            return m_matrix[row][col];
        }

        [[nodiscard]] CUSTOSH_HOST_DEV_MEMBER Matrix<T, Rows, Cols> operator*(const T& scalar) const
        {
            Matrix<T, Rows, Cols> result;

            for (unsigned int i = 0; i < Rows; ++i) {
                for (unsigned int j = 0; j < Cols; ++j) {
                    result(i, j) = scalar * m_matrix[i][j];
                }
            }

            return result;
        }

        [[nodiscard]] CUSTOSH_HOST_DEV_MEMBER friend Matrix<T, Rows, Cols> operator*(const T& scalar, const Matrix& matrix)
        {
            return matrix * scalar;
        }

        [[nodiscard]] CUSTOSH_HOST_DEV_MEMBER Matrix<T, Rows, Cols> operator+(const Matrix<T, Rows, Cols>& other) const
        {
            Matrix<T, Rows, Cols> result;

            for (unsigned int i = 0; i < Rows; ++i) {
                for (unsigned int j = 0; j < Cols; ++j) {
                    result(i, j) = m_matrix[i][j] + other(i, j);
                }
            }

            return result;
        }

        [[nodiscard]] CUSTOSH_HOST_DEV_MEMBER Matrix<T, Rows, Cols> operator-(const Matrix<T, Rows, Cols>& other) const
        {
            return *this + -1 * other;
        }

        template<unsigned int OtherCols>
        [[nodiscard]] CUSTOSH_HOST_DEV_MEMBER Matrix<T, Rows, OtherCols>
        operator*(const Matrix<T, Cols, OtherCols>& other) const
        {
            Matrix<T, Rows, OtherCols> result;

            for (unsigned int i = 0; i < Rows; ++i) {
                for (unsigned int j = 0; j < OtherCols; ++j) {
                    result(i, j) = T();
                    for (unsigned int k = 0; k < Cols; ++k) {
                        result(i, j) += m_matrix[i][k] * other(k, j);
                    }
                }
            }

            return result;
        }

        [[nodiscard]] CUSTOSH_HOST_DEV_MEMBER Matrix<T, Cols, Rows> transpose() const
        {
            Matrix<T, Cols, Rows> result;

            for (unsigned int i = 0; i < Rows; ++i) {
                for (unsigned int j = 0; j < Cols; ++j) {
                    result(j, i) = m_matrix[i][j];
                }
            }

            return result;
        }

        [[nodiscard]] CUSTOSH_HOST_DEV_GETTER unsigned int numRows() const
        { return Rows; }

        [[nodiscard]] CUSTOSH_HOST_DEV_GETTER unsigned int numCols() const
        { return Cols; }

    protected:
        T m_matrix[Rows][Cols];

    }; // Matrix

    template<typename T, unsigned int Size>
    class Vector : public Matrix<T, Size, 1>
    {
    public:
        CUSTOSH_HOST_DEV_MEMBER Vector() : Matrix<T, Size, 1>()
        {
        }

        CUSTOSH_HOST_DEV_MEMBER explicit Vector(Matrix<T, Size, 1> matrix) : Matrix<T, Size, 1>(matrix)
        {
        }

        CUSTOSH_HOST_DEV_MEMBER Vector(const std::initializer_list<T>& init)
        {
            if (init.size() != Size) { CUSTOSH_HOST_DEV_ERR(CUSTOSH_INIT_LIST_ERR_MSG); }

            unsigned int i = 0;
            for (const auto& elem: init) {
                this->m_matrix[i][0] = elem;
                ++i;
            }
        }

        [[nodiscard]] CUSTOSH_HOST_DEV_MEMBER T& operator()(unsigned int idx)
        {
            if (idx >= Size) { CUSTOSH_HOST_DEV_ERR(CUSTOSH_IDX_ERR_MSG); }

            return this->m_matrix[idx][0];
        }

        [[nodiscard]] CUSTOSH_HOST_DEV_MEMBER const T& operator()(unsigned int idx) const
        {
            if (idx >= Size) { CUSTOSH_HOST_DEV_ERR(CUSTOSH_IDX_ERR_MSG); }

            return this->m_matrix[idx][0];
        }

        [[nodiscard]] CUSTOSH_HOST_DEV_MEMBER T dot(const Vector<T, Size>& other) const
        {
            return ((*this).transpose() * other)(0, 0);
        }

        [[nodiscard]] CUSTOSH_HOST_DEV_MEMBER T normSq() const
        {
            T normSq = T();

            for (unsigned int i = 0; i < Size; ++i) {
                normSq += (*this)(i) * (*this)(i);
            }

            return normSq;
        }

        [[nodiscard]] CUSTOSH_HOST_DEV_MEMBER Vector<T, Size> normalized() const
        {
            T normSquared = normSq();

            if (normSquared == 0) { return *this; }

            return Vector<T, Size>(*this * (1.0f / std::sqrt(normSquared)));
        }

    }; // Vector

    template<typename T>
    class Vector2 : public Vector<T, 2>
    {
    public:
        CUSTOSH_HOST_DEV_MEMBER Vector2() : Vector<T, 2>()
        {
        }

        CUSTOSH_HOST_DEV_MEMBER explicit Vector2(Vector<T, 2> vector) : Vector<T, 2>(vector)
        {
        }

        CUSTOSH_HOST_DEV_MEMBER explicit Vector2(Matrix<T, 2, 1> matrix) : Vector<T, 2>(matrix)
        {
        }

        CUSTOSH_HOST_DEV_MEMBER Vector2(const std::initializer_list<T>& init) : Vector<T, 2>(init)
        {
        }

        [[nodiscard]] CUSTOSH_HOST_DEV_GETTER T& x()
        { return (*this)(0); }

        [[nodiscard]] CUSTOSH_HOST_DEV_GETTER const T& x() const
        { return (*this)(0); }

        [[nodiscard]] CUSTOSH_HOST_DEV_GETTER T& y()
        { return (*this)(1); }

        [[nodiscard]] CUSTOSH_HOST_DEV_GETTER const T& y() const
        { return (*this)(1); }

    }; // Vector2

    template<typename T>
    class Vector4 : public Vector<T, 4>
    {
    public:
        CUSTOSH_HOST_DEV_MEMBER Vector4() : Vector<T, 4>()
        {
        }

        CUSTOSH_HOST_DEV_MEMBER explicit Vector4(Vector<T, 4> vector) : Vector<T, 4>(vector)
        {
        }

        CUSTOSH_HOST_DEV_MEMBER explicit Vector4(Matrix<T, 4, 1> matrix) : Vector<T, 4>(matrix)
        {
        }

        CUSTOSH_HOST_DEV_MEMBER Vector4(const std::initializer_list<T>& init) : Vector<T, 4>(init)
        {
        }

        [[nodiscard]] CUSTOSH_HOST_DEV_MEMBER Vector4<T> normalizeW() const
        {
            Vector4<T> result;

            result.x() = this->x() / this->w();
            result.y() = this->y() / this->w();
            result.z() = this->z() / this->w();
            result.w() = 1;

            return result;
        }

        [[nodiscard]] CUSTOSH_HOST_DEV_GETTER T& x()
        { return (*this)(0); }

        [[nodiscard]] CUSTOSH_HOST_DEV_GETTER const T& x() const
        { return (*this)(0); }

        [[nodiscard]] CUSTOSH_HOST_DEV_GETTER T& y()
        { return (*this)(1); }

        [[nodiscard]] CUSTOSH_HOST_DEV_GETTER const T& y() const
        { return (*this)(1); }

        [[nodiscard]] CUSTOSH_HOST_DEV_GETTER T& z()
        { return (*this)(2); }

        [[nodiscard]] CUSTOSH_HOST_DEV_GETTER const T& z() const
        { return (*this)(2); }

        [[nodiscard]] CUSTOSH_HOST_DEV_GETTER T& w()
        { return (*this)(3); }

        [[nodiscard]] CUSTOSH_HOST_DEV_GETTER const T& w() const
        { return (*this)(3); }

    }; // Vector4

    template<typename T>
    class Vector3 : public Vector<T, 3>
    {
    public:
        CUSTOSH_HOST_DEV_MEMBER Vector3() : Vector<T, 3>()
        {
        }

        CUSTOSH_HOST_DEV_MEMBER explicit Vector3(Vector<T, 3> vector) : Vector<T, 3>(vector)
        {
        }

        CUSTOSH_HOST_DEV_MEMBER explicit Vector3(Matrix<T, 3, 1> matrix) : Vector<T, 3>(matrix)
        {
        }

        CUSTOSH_HOST_DEV_MEMBER Vector3(const std::initializer_list<T>& init) : Vector<T, 3>(init)
        {
        }

        [[nodiscard]] CUSTOSH_HOST_DEV_MEMBER Vector3<T> cross(const Vector3<T>& other) const
        {
            Vector3<T> result;

            result(0) = (*this)(1) * other(2) - (*this)(2) * other(1);
            result(1) = (*this)(2) * other(0) - (*this)(0) * other(2);
            result(2) = (*this)(0) * other(1) - (*this)(1) * other(0);

            return result;
        }

        [[nodiscard]] CUSTOSH_HOST_DEV_MEMBER Vector4<T> toHomogeneous() const
        {
            return {x(), y(), z(), static_cast<T>(1)};
        }

        [[nodiscard]] CUSTOSH_HOST_DEV_GETTER T& x()
        { return (*this)(0); }

        [[nodiscard]] CUSTOSH_HOST_DEV_GETTER const T& x() const
        { return (*this)(0); }

        [[nodiscard]] CUSTOSH_HOST_DEV_GETTER T& y()
        { return (*this)(1); }

        [[nodiscard]] CUSTOSH_HOST_DEV_GETTER const T& y() const
        { return (*this)(1); }

        [[nodiscard]] CUSTOSH_HOST_DEV_GETTER T& z()
        { return (*this)(2); }

        [[nodiscard]] CUSTOSH_HOST_DEV_GETTER const T& z() const
        { return (*this)(2); }

    }; // Vector3

    template<typename T>
    class Quaternion
    {
    public:
        CUSTOSH_HOST_DEV_MEMBER Quaternion(T real, Vector3<T> imaginaryVec)
                : m_realPart(real),
                  m_imaginaryVec(std::move(imaginaryVec))
        {
        }

        CUSTOSH_HOST_DEV_MEMBER Quaternion(T real, T i, T j, T k)
                : m_realPart(real), m_imaginaryVec({i, j, k})
        {
        }

        [[nodiscard]] CUSTOSH_HOST_DEV_MEMBER Quaternion<T> operator*(const T& scalar) const
        {
            return {scalar * m_realPart, Vector3<T>(scalar * m_imaginaryVec)};
        }

        [[nodiscard]] CUSTOSH_HOST_DEV_MEMBER friend Quaternion<T> operator*(const T& scalar,
                                                                             const Quaternion<T>& quaternion)
        {
            return quaternion * scalar;
        }

        [[nodiscard]] CUSTOSH_HOST_DEV_MEMBER Quaternion<T> operator+(const Quaternion& other) const
        {
            return {m_realPart + other.m_realPart, m_imaginaryVec + other.m_imaginaryVec};
        }

        [[nodiscard]] CUSTOSH_HOST_DEV_MEMBER Quaternion<T> operator*(const Quaternion& other) const
        {
            return {m_realPart * other.m_realPart - m_imaginaryVec.dot(other.m_imaginaryVec),
                    Vector3<float>(m_realPart * other.m_imaginaryVec +
                                   other.m_realPart * m_imaginaryVec +
                                   m_imaginaryVec.cross(other.m_imaginaryVec))};
        }

        [[nodiscard]] CUSTOSH_HOST_DEV_MEMBER Quaternion<T> conjunction() const
        {
            return {m_realPart, Vector3<T>(-1 * m_imaginaryVec)};
        }

        [[nodiscard]] CUSTOSH_HOST_DEV_MEMBER T normSq() const
        {
            return m_realPart * m_realPart + m_imaginaryVec.normSq();
        }

        [[nodiscard]] CUSTOSH_HOST_DEV_MEMBER Quaternion normalized() const
        {
            return (*this) * (1.f / sqrt(this->normSq()));
        }

        [[nodiscard]] CUSTOSH_HOST_DEV_GETTER T realPart() const
        { return m_realPart; }

        [[nodiscard]] CUSTOSH_HOST_DEV_GETTER Vector3<T> imaginaryVec() const
        { return m_imaginaryVec; }

    private:
        T m_realPart;
        Vector3<T> m_imaginaryVec;

    }; // Quaternion

    class PerspectiveMatrix : public Matrix<float, 4, 4>
    {
    public:
        CUSTOSH_HOST_DEV_MEMBER PerspectiveMatrix(float nearPlaneDist, float farPlaneDist)
                : Matrix<float, 4, 4>(init(nearPlaneDist, farPlaneDist))
        {
        }

    private:
        CUSTOSH_HOST_DEV_MEMBER static Matrix<float, 4, 4> init(float nearPlaneDist, float farPlaneDist)
        {
            return {{nearPlaneDist, 0.f,           0.f,                          0.f},
                    {0.f,           nearPlaneDist, 0.f,                          0.f},
                    {0.f,           0.f,           nearPlaneDist + farPlaneDist, -nearPlaneDist * farPlaneDist},
                    {0.f,           0.f,           1.f,                          0.f}};
        }

    }; // PerspectiveMatrix

    class TranslationMatrix : public Matrix<float, 4, 4>
    {
    public:
        CUSTOSH_HOST_DEV_MEMBER explicit TranslationMatrix(const Vector3<float>& translationVec)
                : Matrix<float, 4, 4>(init(translationVec))
        {
        }

    private:
        CUSTOSH_HOST_DEV_MEMBER static Matrix<float, 4, 4> init(const Vector3<float>& translationVec)
        {
            return {{1.f, 0.f, 0.f, translationVec.x()},
                    {0.f, 1.f, 0.f, translationVec.y()},
                    {0.f, 0.f, 1.f, translationVec.z()},
                    {0.f, 0.f, 0.f, 1.f}};
        }

    }; // TranslationMatrix

    class ScalingMatrix : public Matrix<float, 4, 4>
    {
    public:
        CUSTOSH_HOST_DEV_MEMBER explicit ScalingMatrix(const Vector3<float>& scalingVec)
                : Matrix<float, 4, 4>(init(scalingVec))
        {
        }

    private:
        CUSTOSH_HOST_DEV_MEMBER static Matrix<float, 4, 4> init(const Vector3<float>& scalingVec)
        {
            return {{scalingVec.x(), 0.f,            0.f,            0.f},
                    {0.f,            scalingVec.y(), 0.f,            0.f},
                    {0.f,            0.f,            scalingVec.z(), 0.f},
                    {0.f,            0.f,            0.f,            1.f}};
        }

    }; // ScalingMatrix

    class RotationMatrix : public Matrix<float, 4, 4>
    {
    public:
        CUSTOSH_HOST_DEV_MEMBER explicit RotationMatrix(const Quaternion<float>& rotationQuaternion)
                : Matrix<float, 4, 4>(init(rotationQuaternion.normalized()))
        {
        }

        CUSTOSH_HOST_DEV_MEMBER RotationMatrix(const Vector3<float>& rotationVec, float angle)
                : Matrix<float, 4, 4>(init({cos(angle / 2),
                                            Vector3<float>(sin(angle / 2) * rotationVec.normalized())}))
        {
        }

    private:
        CUSTOSH_HOST_DEV_MEMBER static Matrix<float, 4, 4> init(const Quaternion<float>& rotationQuaternion)
        {
            float w = rotationQuaternion.realPart();
            float x = rotationQuaternion.imaginaryVec().x();
            float y = rotationQuaternion.imaginaryVec().y();
            float z = rotationQuaternion.imaginaryVec().z();

            return {{1.f - 2.f * (y * y + z * z), 2.f * (x * y - w * z),       2.f * (x * z + w * y),       0.f},
                    {2.f * (x * y + w * z),       1.f - 2.f * (x * x + z * z), 2.f * (y * z - w * x),       0.f},
                    {2.f * (x * z - w * y),       2.f * (y * z + w * x),       1.f - 2.f * (x * x + y * y), 0.f},
                    {0.f,                         0.f,                         0.f,                         1.f}};
        }

    }; // RotationMatrix

    class DecentralizedTransformMatrix : public Matrix<float, 4, 4>
    {
    public:
        CUSTOSH_HOST_DEV_MEMBER DecentralizedTransformMatrix(const Matrix<float, 4, 4>& transformMat,
                                                             const Vector3<float>& origin)
                : Matrix<float, 4, 4>(init(transformMat, origin))
        {
        }

    private:
        CUSTOSH_HOST_DEV_MEMBER static Matrix<float, 4, 4> init(const Matrix<float, 4, 4>& transformMat,
                                                                const Vector3<float>& origin)
        {
            auto centerTranslationMat = TranslationMatrix(Vector3<float>(origin * -1));
            auto originTranslationMat = TranslationMatrix(origin);

            return originTranslationMat * transformMat * centerTranslationMat;
        }

    }; // DecentralizedRotationMatrix

    class OrtProjMatrix : public Matrix<float, 4, 4>
    {
    public:
        CUSTOSH_HOST_DEV_MEMBER OrtProjMatrix(const Vector3<float>& fromMinCorner,
                                              const Vector3<float>& fromMaxCorner,
                                              const Vector3<float>& toMinCorner,
                                              const Vector3<float>& toMaxCorner)
                : Matrix<float, 4, 4>(init(fromMinCorner,
                                           fromMaxCorner,
                                           toMinCorner,
                                           toMaxCorner))
        {
        }

    private:
        CUSTOSH_HOST_DEV_MEMBER static Matrix<float, 4, 4> init(const Vector3<float>& fromMinCorner,
                                                                const Vector3<float>& fromMaxCorner,
                                                                const Vector3<float>& toMinCorner,
                                                                const Vector3<float>& toMaxCorner)
        {
            auto centerTranslationVec = Vector3<float>(boxCenter(fromMinCorner, fromMaxCorner) * -1);
            TranslationMatrix centerTranslationMatrix(centerTranslationVec);

            Vector3<float> toTranslationVec = boxCenter(toMinCorner, toMaxCorner);
            TranslationMatrix toTranslationMatrix(toTranslationVec);

            auto scalingVecAuxTo = Vector3<float>(toMaxCorner - toMinCorner);
            auto scalingVecAuxFrom = Vector3<float>(fromMaxCorner - fromMinCorner);

            Vector3<float> scalingVecAux = {scalingVecAuxTo.x() / scalingVecAuxFrom.x(),
                                            scalingVecAuxTo.y() / scalingVecAuxFrom.y(),
                                            scalingVecAuxTo.z() / scalingVecAuxFrom.z()};

            ScalingMatrix scalingMatrix(scalingVecAux);

            return toTranslationMatrix * scalingMatrix * centerTranslationMatrix;
        }

        CUSTOSH_HOST_DEV_MEMBER static Vector3<float> boxCenter(const Vector3<float>& minCorner,
                                                                const Vector3<float>& maxCorner)
        {
            return Vector3<float>((minCorner + maxCorner) * 0.5f);
        }

    }; // OrtProjMatrix

    class PerspectiveProjMatrix : public Matrix<float, 4, 4>
    {
    public:
        CUSTOSH_HOST_DEV_MEMBER PerspectiveProjMatrix(const PerspectiveMatrix& pm, const OrtProjMatrix& opm)
                : Matrix<float, 4, 4>(opm * pm)
        {
        }

    }; // PerspectiveProjMatrix

    /* Aliases */
    using Vertex2D = Vector2<float>;
    using Vertex3D = Vector3<float>;
    using TransformMatrix = Matrix<float, 4, 4>;

    /* Structs */
    struct triangleIndices_t
    {
        unsigned int p0;
        unsigned int p1;
        unsigned int p2;

        CUSTOSH_HOST_DEV_MEMBER explicit triangleIndices_t(
                unsigned int p0 = 0,
                unsigned int p1 = 0,
                unsigned int p2 = 0
        ) : p0(p0), p1(p1), p2(p2)
        {
        }
    };

    struct lightSource_t
    {
        Vertex3D coords;
        float intensity;

        CUSTOSH_HOST_DEV_MEMBER explicit lightSource_t(
                const Vertex3D& coords = Vertex3D(),
                float intensity = 1.f
        ) : coords(coords), intensity(intensity)
        {
        }
    };

} // custosh

#undef CUSTOSH_IF_CUDACC
#undef CUSTOSH_HOST_DEV_ERR
#undef CUSTOSH_INIT_LIST_ERR_MSG
#undef CUSTOSH_IDX_ERR_MSG
#undef CUSTOSH_HOST_DEV_AUX_FUNC
#undef CUSTOSH_HOST_DEV_MEMBER
#undef CUSTOSH_HOST_DEV_GETTER


#endif // CUSTOSH_UTILITY_H
