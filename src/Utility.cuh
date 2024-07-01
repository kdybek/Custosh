#ifndef CUSTOSH_UTILITY_CUH
#define CUSTOSH_UTILITY_CUH


#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <cmath>
#include <numbers>
#include <cuda_runtime.h>

#include "CustoshExcept.h"

#define CUDA_CHECK(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::string errMsg = "CUDA error at "; \
        errMsg += __FILE__; \
        errMsg += ":"; \
        errMsg += std::to_string(__LINE__); \
        errMsg += " - "; \
        errMsg += cudaGetErrorString(error); \
        throw CustoshException(errMsg); \
    } \
} while(0)

namespace Custosh
{
    inline const std::string ASCIIByBrightness =
            R"( .'`,_^"-+:;!><~?iI[]{}1()|\/tfjrnxuvczXYUJCLQ0OZmwqpdkbhao*#MW&8%B@$)";

    template<typename T, unsigned int Rows, unsigned int Cols>
    class Matrix
    {
    public:
        __host__ __device__ Matrix()
        {
            for (unsigned int i = 0; i < Rows; ++i) {
                for (unsigned int j = 0; j < Cols; ++j) {
                    m_matrix[i][j] = T();
                }
            }
        }

        __host__ __device__ Matrix(const std::initializer_list<std::initializer_list<T>>& init)
        {
            unsigned int i = 0;
            for (const auto& row: init) {
                unsigned int j = 0;
                for (const auto& elem: row) {
                    m_matrix[i][j] = elem;
                    ++j;
                }
                ++i;
            }
        }

        __host__ __device__ T& operator()(unsigned int row, unsigned int col)
        {
            return m_matrix[row][col];
        }

        __host__ __device__ const T& operator()(unsigned int row, unsigned int col) const
        {
            return m_matrix[row][col];
        }

        [[nodiscard]] __host__ __device__ unsigned int getNRows() const
        {
            return Rows;
        }

        [[nodiscard]] __host__ __device__ unsigned int getNCols() const
        {
            return Cols;
        }

        __host__ __device__ Matrix<T, Rows, Cols> operator*(const T& scalar) const
        {
            Matrix<T, Rows, Cols> result;

            for (unsigned int i = 0; i < Rows; ++i) {
                for (unsigned int j = 0; j < Cols; ++j) {
                    result(i, j) = scalar * m_matrix[i][j];
                }
            }

            return result;
        }

        __host__ __device__ friend Matrix<T, Rows, Cols> operator*(const T& scalar, const Matrix& matrix)
        {
            return matrix * scalar;
        }

        __host__ __device__ Matrix<T, Rows, Cols> operator+(const Matrix<T, Rows, Cols>& other) const
        {
            Matrix<T, Rows, Cols> result;

            for (unsigned int i = 0; i < Rows; ++i) {
                for (unsigned int j = 0; j < Cols; ++j) {
                    result(i, j) = m_matrix[i][j] + other(i, j);
                }
            }

            return result;
        }

        __host__ __device__ Matrix<T, Rows, Cols> operator-(const Matrix<T, Rows, Cols>& other) const
        {
            return *this + -1 * other;
        }

        template<unsigned int OtherCols>
        __host__ __device__ Matrix<T, Rows, OtherCols> operator*(const Matrix<T, Cols, OtherCols>& other) const
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

        [[nodiscard]] __host__ __device__ Matrix<T, Cols, Rows> transpose() const
        {
            Matrix<T, Cols, Rows> result;

            for (unsigned int i = 0; i < Rows; ++i) {
                for (unsigned int j = 0; j < Cols; ++j) {
                    result(j, i) = m_matrix[i][j];
                }
            }

            return result;
        }

    protected:
        T m_matrix[Rows][Cols];

    }; // Matrix

    template<typename T, unsigned int Size>
    class Vector : public Matrix<T, Size, 1>
    {
    public:
        __host__ __device__ Vector() : Matrix<T, Size, 1>()
        {
        }

        __host__ __device__ explicit Vector(Matrix<T, Size, 1> matrix) : Matrix<T, Size, 1>(matrix)
        {
        }

        __host__ __device__ Vector(const std::initializer_list<T>& init)
        {
            unsigned int i = 0;
            for (const auto& elem: init) {
                this->m_matrix[i][0] = elem;
                ++i;
            }
        }

        __host__ __device__ T& operator()(unsigned int index)
        {
            return this->m_matrix[index][0];
        }

        __host__ __device__ const T& operator()(unsigned int index) const
        {
            return this->m_matrix[index][0];
        }

        [[nodiscard]] __host__ __device__ T dot(const Vector<T, Size>& other) const
        {
            return ((*this).transpose() * other)(0, 0);
        }

        [[nodiscard]] __host__ __device__ T normSq() const
        {
            T normSq = T();

            for (unsigned int i = 0; i < Size; ++i) {
                normSq += (*this)(i) * (*this)(i);
            }

            return normSq;
        }

        [[nodiscard]] __host__ __device__ Vector<T, Size> normalized() const
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
        __host__ __device__ Vector2() : Vector<T, 2>()
        {
        }

        __host__ __device__ explicit Vector2(Vector<T, 2> vector) : Vector<T, 2>(vector)
        {
        }

        __host__ __device__ explicit Vector2(Matrix<T, 2, 1> matrix) : Vector<T, 2>(matrix)
        {
        }

        __host__ __device__ Vector2(const std::initializer_list<T>& init) : Vector<T, 2>(init)
        {
        }

        __host__ __device__ T& x()
        {
            return (*this)(0);
        }

        [[nodiscard]] __host__ __device__ const T& x() const
        {
            return (*this)(0);
        }

        __host__ __device__ T& y()
        {
            return (*this)(1);
        }

        [[nodiscard]] __host__ __device__ const T& y() const
        {
            return (*this)(1);
        }

    }; // Vector2

    template<typename T>
    class Vector4 : public Vector<T, 4>
    {
    public:
        __host__ __device__ Vector4() : Vector<T, 4>()
        {
        }

        __host__ __device__ explicit Vector4(Vector<T, 4> vector) : Vector<T, 4>(vector)
        {
        }

        __host__ __device__ explicit Vector4(Matrix<T, 4, 1> matrix) : Vector<T, 4>(matrix)
        {
        }

        __host__ __device__ Vector4(const std::initializer_list<T>& init) : Vector<T, 4>(init)
        {
        }

        __host__ __device__ T& x()
        {
            return (*this)(0);
        }

        [[nodiscard]] __host__ __device__ const T& x() const
        {
            return (*this)(0);
        }

        __host__ __device__ T& y()
        {
            return (*this)(1);
        }

        [[nodiscard]] __host__ __device__ const T& y() const
        {
            return (*this)(1);
        }

        __host__ __device__ T& z()
        {
            return (*this)(2);
        }

        [[nodiscard]] __host__ __device__ const T& z() const
        {
            return (*this)(2);
        }

        __host__ __device__ T& w()
        {
            return (*this)(3);
        }

        [[nodiscard]] __host__ __device__ const T& w() const
        {
            return (*this)(3);
        }

        [[nodiscard]] __host__ __device__ Vector4<T> normalizeW() const
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
        __host__ __device__ Vector3() : Vector<T, 3>()
        {
        }

        __host__ __device__ explicit Vector3(Vector<T, 3> vector) : Vector<T, 3>(vector)
        {
        }

        __host__ __device__ explicit Vector3(Matrix<T, 3, 1> matrix) : Vector<T, 3>(matrix)
        {
        }

        __host__ __device__ Vector3(const std::initializer_list<T>& init) : Vector<T, 3>(init)
        {
        }

        __host__ __device__ T& x()
        {
            return (*this)(0);
        }

        [[nodiscard]] __host__ __device__ const T& x() const
        {
            return (*this)(0);
        }

        __host__ __device__ T& y()
        {
            return (*this)(1);
        }

        [[nodiscard]] __host__ __device__ const T& y() const
        {
            return (*this)(1);
        }

        __host__ __device__ T& z()
        {
            return (*this)(2);
        }

        [[nodiscard]] __host__ __device__ const T& z() const
        {
            return (*this)(2);
        }

        [[nodiscard]] __host__ __device__ Vector3<T> cross(const Vector3<T>& other) const
        {
            Vector3<T> result;

            result(0) = (*this)(1) * other(2) - (*this)(2) * other(1);
            result(1) = (*this)(2) * other(0) - (*this)(0) * other(2);
            result(2) = (*this)(0) * other(1) - (*this)(1) * other(0);

            return result;
        }

        [[nodiscard]] __host__ __device__ Vector4<T> toHomogeneous() const
        {
            return {x(), y(), z(), 1};
        }

    }; // Vector3

    class PerspectiveMatrix : public Matrix<float, 4, 4>
    {
    public:
        __host__ __device__ PerspectiveMatrix(float nearPlaneDist, float farPlaneDist)
                : Matrix<float, 4, 4>(init(nearPlaneDist, farPlaneDist))
        {
        }

    private:
        __host__ __device__ static Matrix<float, 4, 4> init(float nearPlaneDist, float farPlaneDist)
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
        __host__ __device__ explicit TranslationMatrix(const Vector3<float>& translationVec)
                : Matrix<float, 4, 4>(init(translationVec))
        {
        }

    private:
        __host__ __device__ static Matrix<float, 4, 4> init(const Vector3<float>& translationVec)
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
        __host__ __device__ explicit ScalingMatrix(const Vector3<float>& scalingVec)
                : Matrix<float, 4, 4>(init(scalingVec))
        {
        }

    private:
        __host__ __device__ static Matrix<float, 4, 4> init(const Vector3<float>& scalingVec)
        {
            return {{scalingVec.x(), 0.f,            0.f,            0.f},
                    {0.f,            scalingVec.y(), 0.f,            0.f},
                    {0.f,            0.f,            scalingVec.z(), 0.f},
                    {0.f,            0.f,            0.f,            1.f}};
        }

    }; // ScalingMatrix

    class OrtProjMatrix : public Matrix<float, 4, 4>
    {
    public:
        __host__ __device__ OrtProjMatrix(const Vector3<float>& fromMinCorner,
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
        __host__ __device__ static Matrix<float, 4, 4> init(const Vector3<float>& fromMinCorner,
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

        __host__ __device__ static Vector3<float> boxCenter(const Vector3<float>& minCorner,
                                                            const Vector3<float>& maxCorner)
        {
            return Vector3<float>((minCorner + maxCorner) * 0.5f);
        }

    }; // OrtProjMatrix

    class PerspectiveProjMatrix : public Matrix<float, 4, 4>
    {
    public:
        __host__ __device__ PerspectiveProjMatrix(const PerspectiveMatrix& pm, const OrtProjMatrix& opm)
                : Matrix<float, 4, 4>(opm * pm)
        {
        }

    }; // PerspectiveProjMatrix

    template<typename T>
    class HostDevResizableMatrix
    {
    public:
        __host__ HostDevResizableMatrix(unsigned int rows, unsigned int cols)
                : m_rows(rows),
                  m_cols(cols),
                  m_hostArrPtr(new T[rows * cols]),
                  m_devArrPtr(nullptr)
        {
            CUDA_CHECK(cudaMalloc(&m_devArrPtr, rows * cols * sizeof(T)));
        }

        __host__ ~HostDevResizableMatrix()
        {
            delete[] m_hostArrPtr;
            cudaFree(m_devArrPtr);
        }

        __host__ __device__ HostDevResizableMatrix(const HostDevResizableMatrix&) = delete;

        __host__ __device__ HostDevResizableMatrix& operator=(const HostDevResizableMatrix&) = delete;

        // The data in the matrix gets lost when resizing.
        __host__ void resizeAndClean(unsigned int newRows, unsigned int newCols)
        {
            delete[] m_hostArrPtr;
            CUDA_CHECK(cudaFree(m_devArrPtr));

            m_hostArrPtr = new T[newRows * newCols];
            CUDA_CHECK(cudaMalloc(&m_devArrPtr, newRows * newCols * sizeof(T)));

            m_rows = newRows;
            m_cols = newCols;
        }

        __host__ void loadToHost() const
        {
            CUDA_CHECK(cudaMemcpy(m_hostArrPtr,
                                  m_devArrPtr,
                                  m_rows * m_cols * sizeof(T),
                                  cudaMemcpyDeviceToHost));
        }

        __host__ void loadToDev() const
        {
            CUDA_CHECK(cudaMemcpy(m_devArrPtr,
                                  m_hostArrPtr,
                                  m_rows * m_cols * sizeof(T),
                                  cudaMemcpyHostToDevice));
        }

        __host__ const T* devData() const
        {
            return m_devArrPtr;
        }

        __host__ T* devData()
        {
            return m_devArrPtr;
        }

        __host__ __device__ T& operator()(unsigned int row, unsigned int col)
        {
#ifdef __CUDA_ARCH__
            if (row >= m_rows || col >= m_cols) {
                printf("Index out of bounds: row = %u, col = %u\n", row, col);
                asm("trap;");
            }
            return m_devArrPtr[m_cols * row + col];
#else
            if (row >= m_rows || col >= m_cols) {
                throw CustoshException("Index out of bounds");
            }
            return m_hostArrPtr[m_cols * row + col];
#endif
        }

        __host__ __device__ const T& operator()(unsigned int row, unsigned int col) const
        {
#ifdef __CUDA_ARCH__
            if (row >= m_rows || col >= m_cols) {
                printf("Index out of bounds: row = %u, col = %u\n", row, col);
                asm("trap;");
            }
            return m_devArrPtr[m_cols * row + col];
#else
            if (row >= m_rows || col >= m_cols) {
                throw CustoshException("Index out of bounds");
            }
            return m_hostArrPtr[m_cols * row + col];
#endif
        }

        [[nodiscard]] __host__ __device__ unsigned int getNRows() const
        {
            return m_rows;
        }

        [[nodiscard]] __host__ __device__ unsigned int getNCols() const
        {
            return m_cols;
        }

    protected:
        unsigned int m_rows;
        unsigned int m_cols;
        T* m_hostArrPtr;
        T* m_devArrPtr;

    }; // HostDevResizableMatrix

    class BrightnessMap : public HostDevResizableMatrix<float>
    {
    public:
        __host__ BrightnessMap(unsigned int rows, unsigned int cols)
                : HostDevResizableMatrix<float>(rows, cols)
        {
        }

        [[nodiscard]] __host__ std::string rowToString(unsigned int row) const
        {
            std::string buffer;

            for (unsigned int j = 0; j < m_cols; ++j) {
                buffer += brightnessToASCII((*this)(row, j));
            }

            return buffer;
        }

    private:
        __host__ static char brightnessToASCII(float brightness)
        {
            unsigned int idx = std::ceil(brightness * static_cast<float>(ASCIIByBrightness.size() - 1));
            return ASCIIByBrightness.at(idx);
        }

    }; // BrightnessMap

    template<typename T>
    class Quaternion
    {
    public:
        __host__ __device__ Quaternion(T real, Vector3<T> imaginaryVec)
                : m_realPart(real),
                  m_imaginaryVec(std::move(imaginaryVec))
        {
        }

        __host__ __device__ Quaternion(T real, T i, T j, T k)
                : m_realPart(real), m_imaginaryVec({i, j, k})
        {
        }

        [[nodiscard]] __host__ __device__ T getRealPart() const
        {
            return m_realPart;
        }

        [[nodiscard]] __host__ __device__ Vector3<T> getImaginaryVec() const
        {
            return m_imaginaryVec;
        }

        __host__ __device__ Quaternion<T> operator*(const T& scalar) const
        {
            return {scalar * m_realPart, Vector3<T>(scalar * m_imaginaryVec)};
        }

        __host__ __device__ friend Quaternion<T> operator*(const T& scalar, const Quaternion<T>& quaternion)
        {
            return quaternion * scalar;
        }

        __host__ __device__ Quaternion<T> operator+(const Quaternion& other) const
        {
            return {m_realPart + other.m_realPart, m_imaginaryVec + other.m_imaginaryVec};
        }

        __host__ __device__ Quaternion<T> operator*(const Quaternion& other) const
        {
            return {m_realPart * other.m_realPart - m_imaginaryVec.dot(other.m_imaginaryVec),
                    Vector3<float>(m_realPart * other.m_imaginaryVec +
                                   other.m_realPart * m_imaginaryVec +
                                   m_imaginaryVec.cross(other.m_imaginaryVec))};
        }

        [[nodiscard]] __host__ __device__ Quaternion<T> conjunction() const
        {
            return {m_realPart, Vector3<T>(-1 * m_imaginaryVec)};
        }

        [[nodiscard]] __host__ __device__ T normSq() const
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

        __host__ __device__ explicit triangle3D_t(
                const Vector3<float>& p0 = Vector3<float>(),
                const Vector3<float>& p1 = Vector3<float>(),
                const Vector3<float>& p2 = Vector3<float>()
        ) : p0(p0), p1(p1), p2(p2)
        {
        }
    };

    struct triangle2D_t
    {
        Vector2<float> p0;
        Vector2<float> p1;
        Vector2<float> p2;

        __host__ __device__ explicit triangle2D_t(
                const Vector2<float>& p0 = Vector2<float>(),
                const Vector2<float>& p1 = Vector2<float>(),
                const Vector2<float>& p2 = Vector2<float>()
        ) : p0(p0), p1(p1), p2(p2)
        {
        }
    };

    struct triangleIndices_t
    {
        unsigned int p0;
        unsigned int p1;
        unsigned int p2;

        __host__ __device__ explicit triangleIndices_t(
                unsigned int p0 = 0,
                unsigned int p1 = 0,
                unsigned int p2 = 0
        ) : p0(p0), p1(p1), p2(p2)
        {
        }
    };

    struct boundingBox_t
    {
        int xMax;
        int xMin;
        int yMax;
        int yMin;

        __host__ __device__ explicit boundingBox_t(
                int xMax = 0,
                int xMin = 0,
                int yMax = 0,
                int yMin = 0
        ) : xMax(xMax), xMin(xMin), yMax(yMax), yMin(yMin)
        {
        }
    };

    struct barycentricCoords_t
    {
        float alpha;
        float beta;
        float gamma;

        __host__ __device__ explicit barycentricCoords_t(
                float alpha = 0.f,
                float beta = 0.f,
                float gamma = 0.f
        ) : alpha(alpha), beta(beta), gamma(gamma)
        {
        }
    };

    struct pixel_t
    {
        bool occupied;
        Vector3<float> coords;
        Vector3<float> normal;

        __host__ __device__ explicit pixel_t(
                bool occupied = false,
                const Vector3<float>& coords = Vector3<float>(),
                const Vector3<float>& normal = Vector3<float>()
        ) : occupied(occupied), coords(coords), normal(normal)
        {
        }
    };

    struct lightSource_t
    {
        Vector3<float> coords;
        float intensity;

        __host__ __device__ explicit lightSource_t(
                const Vector3<float>& coords = Vector3<float>(),
                float intensity = 1.f
        ) : coords(coords), intensity(intensity)
        {
        }
    };

    __host__ __device__ inline float degreesToRadians(float degrees)
    {
        return degrees * (std::numbers::pi_v<float> / 180.f);
    }

    template<typename T>
    __host__ __device__ inline T clamp(T a, T min, T max)
    {
        if (a < min) { return min; }
        else if (a > max) { return max; }
        else { return a; }
    }

    template<typename T>
    __host__ __device__ inline T max(T a, T b)
    {
        return a < b ? b : a;
    }

} // Custosh


#endif // CUSTOSH_UTILITY_CUH
