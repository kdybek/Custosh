#ifndef CUSTOSH_UTILITY_CUH
#define CUSTOSH_UTILITY_CUH


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

#ifdef __CUDA_ARCH__
#define HOST_DEV_ERR(message) \
    do { \
        printf(message); \
        asm("trap;"); \
    } while(0)
#else
#define HOST_DEV_ERR(message) \
    do { \
        throw CustoshException(message); \
    } while(0)
#endif

#define INIT_LIST_ERR_MSG "incorrect initializer list"
#define IDX_ERR_MSG "index out of bounds"

#define DARR_BASE_SIZE 8

#define HOST_DEV_AUX_FUNC __host__ __device__ inline constexpr

namespace Custosh
{
    /* Functions */
    [[nodiscard]] HOST_DEV_AUX_FUNC float degreesToRadians(float degrees)
    {
        return degrees * (std::numbers::pi_v<float> / 180.f);
    }

    template<typename T>
    [[nodiscard]] HOST_DEV_AUX_FUNC T clamp(T a, T min, T max)
    {
        if (a < min) { return min; }
        else if (a > max) { return max; }
        else { return a; }
    }

    template<typename T>
    [[nodiscard]] HOST_DEV_AUX_FUNC T max3(T a, T b, T c)
    {
        return max(max(a, b), c);
    }

    template<typename T>
    [[nodiscard]] HOST_DEV_AUX_FUNC T min3(T a, T b, T c)
    {
        return min(min(a, b), c);
    }

    template<typename T>
    HOST_DEV_AUX_FUNC void swap(T& a, T& b)
    {
        T temp = a;
        a = b;
        b = temp;
    }

    /* Classes */
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
            if (init.size() != Rows) { HOST_DEV_ERR(INIT_LIST_ERR_MSG); }

            unsigned int i = 0;
            for (const auto& row: init) {
                if (row.size() != Cols) { HOST_DEV_ERR(INIT_LIST_ERR_MSG); }

                unsigned int j = 0;
                for (const auto& elem: row) {
                    m_matrix[i][j] = elem;
                    ++j;
                }
                ++i;
            }
        }

        [[nodiscard]] __host__ __device__ T& operator()(unsigned int row, unsigned int col)
        {
            if (row >= Rows || col >= Cols) { HOST_DEV_ERR(IDX_ERR_MSG); }

            return m_matrix[row][col];
        }

        [[nodiscard]] __host__ __device__ const T& operator()(unsigned int row, unsigned int col) const
        {
            if (row >= Rows || col >= Cols) { HOST_DEV_ERR(IDX_ERR_MSG); }

            return m_matrix[row][col];
        }

        [[nodiscard]] __host__ __device__ Matrix<T, Rows, Cols> operator*(const T& scalar) const
        {
            Matrix<T, Rows, Cols> result;

            for (unsigned int i = 0; i < Rows; ++i) {
                for (unsigned int j = 0; j < Cols; ++j) {
                    result(i, j) = scalar * m_matrix[i][j];
                }
            }

            return result;
        }

        [[nodiscard]] __host__ __device__ friend Matrix<T, Rows, Cols> operator*(const T& scalar, const Matrix& matrix)
        {
            return matrix * scalar;
        }

        [[nodiscard]] __host__ __device__ Matrix<T, Rows, Cols> operator+(const Matrix<T, Rows, Cols>& other) const
        {
            Matrix<T, Rows, Cols> result;

            for (unsigned int i = 0; i < Rows; ++i) {
                for (unsigned int j = 0; j < Cols; ++j) {
                    result(i, j) = m_matrix[i][j] + other(i, j);
                }
            }

            return result;
        }

        [[nodiscard]] __host__ __device__ Matrix<T, Rows, Cols> operator-(const Matrix<T, Rows, Cols>& other) const
        {
            return *this + -1 * other;
        }

        template<unsigned int OtherCols>
        [[nodiscard]] __host__ __device__ Matrix<T, Rows, OtherCols>
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

        [[nodiscard]] __host__ __device__ unsigned int numRows() const
        { return Rows; }

        [[nodiscard]] __host__ __device__ unsigned int numCols() const
        { return Cols; }

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
            if (init.size() != Size) { HOST_DEV_ERR(INIT_LIST_ERR_MSG); }

            unsigned int i = 0;
            for (const auto& elem: init) {
                this->m_matrix[i][0] = elem;
                ++i;
            }
        }

        [[nodiscard]] __host__ __device__ T& operator()(unsigned int idx)
        {
            if (idx >= Size) { HOST_DEV_ERR(IDX_ERR_MSG); }

            return this->m_matrix[idx][0];
        }

        [[nodiscard]] __host__ __device__ const T& operator()(unsigned int idx) const
        {
            if (idx >= Size) { HOST_DEV_ERR(IDX_ERR_MSG); }

            return this->m_matrix[idx][0];
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

        [[nodiscard]] __host__ __device__ T& x()
        { return (*this)(0); }

        [[nodiscard]] __host__ __device__ const T& x() const
        { return (*this)(0); }

        [[nodiscard]] __host__ __device__ T& y()
        { return (*this)(1); }

        [[nodiscard]] __host__ __device__ const T& y() const
        { return (*this)(1); }

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

        [[nodiscard]] __host__ __device__ Vector4<T> normalizeW() const
        {
            Vector4<T> result;

            result.x() = this->x() / this->w();
            result.y() = this->y() / this->w();
            result.z() = this->z() / this->w();
            result.w() = 1;

            return result;
        }

        [[nodiscard]] __host__ __device__ T& x()
        { return (*this)(0); }

        [[nodiscard]] __host__ __device__ const T& x() const
        { return (*this)(0); }

        [[nodiscard]] __host__ __device__ T& y()
        { return (*this)(1); }

        [[nodiscard]] __host__ __device__ const T& y() const
        { return (*this)(1); }

        [[nodiscard]] __host__ __device__ T& z()
        { return (*this)(2); }

        [[nodiscard]] __host__ __device__ const T& z() const
        { return (*this)(2); }

        [[nodiscard]] __host__ __device__ T& w()
        { return (*this)(3); }

        [[nodiscard]] __host__ __device__ const T& w() const
        { return (*this)(3); }

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
            return {x(), y(), z(), static_cast<T>(1)};
        }

        [[nodiscard]] __host__ __device__ T& x()
        { return (*this)(0); }

        [[nodiscard]] __host__ __device__ const T& x() const
        { return (*this)(0); }

        [[nodiscard]] __host__ __device__ T& y()
        { return (*this)(1); }

        [[nodiscard]] __host__ __device__ const T& y() const
        { return (*this)(1); }

        [[nodiscard]] __host__ __device__ T& z()
        { return (*this)(2); }

        [[nodiscard]] __host__ __device__ const T& z() const
        { return (*this)(2); }

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
    class DevPtr
    {
    public:
        __host__ explicit DevPtr(unsigned int size) : m_devPtr(nullptr), m_size(size)
        {
            CUDA_CHECK(cudaMalloc(&m_devPtr, size * sizeof(T)));
        }

        __host__ virtual ~DevPtr()
        {
            if (m_devPtr) { cudaFree(m_devPtr); }
        }

        __host__ __device__ DevPtr(const DevPtr&) = delete;

        __host__ __device__ DevPtr& operator=(const DevPtr&) = delete;

        __host__ DevPtr(DevPtr&& other) noexcept: m_devPtr(other.m_devPtr), m_size(other.m_size)
        {
            other.m_devPtr = nullptr;
            other.m_size = 0;
        }

        __host__ DevPtr& operator=(DevPtr&& other) noexcept
        {
            if (this != &other) {
                if (m_devPtr) { cudaFree(m_devPtr); }

                m_devPtr = other.m_devPtr;
                m_size = other.m_size;

                other.m_devPtr = nullptr;
                other.m_size = 0;
            }

            return *this;
        }

        __host__ void resizeAndDiscardData(unsigned int newSize)
        {
            if (m_devPtr) { CUDA_CHECK(cudaFree(m_devPtr)); }

            CUDA_CHECK(cudaMalloc(&m_devPtr, newSize * sizeof(T)));
            m_size = newSize;
        }

        __host__ void resizeAndCopy(unsigned int newSize)
        {
            T* oldDevPtr = m_devPtr;
            unsigned int oldSize = m_size;

            CUDA_CHECK(cudaMalloc(&m_devPtr, newSize * sizeof(T)));
            m_size = newSize;

            unsigned int sizeToCopy = std::min(oldSize, newSize);

            if (oldDevPtr) {
                CUDA_CHECK(cudaMemcpy(m_devPtr, oldDevPtr, sizeToCopy * sizeof(T), cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaFree(oldDevPtr));
            }
        }

        [[nodiscard]] __host__ T* get() const
        { return m_devPtr; }

        [[nodiscard]] __host__ unsigned int size() const
        { return m_size; }

    private:
        T* m_devPtr;
        unsigned int m_size;

    }; // DevPtr

    template<typename T>
    class HostDevPtr
    {
    public:
        __host__ explicit HostDevPtr(unsigned int size)
                : m_hostPtr(nullptr),
                  m_devPtr(nullptr),
                  m_size(size)
        {
            CUDA_CHECK(cudaMallocHost(&m_hostPtr, size * sizeof(T)));
            CUDA_CHECK(cudaMalloc(&m_devPtr, size * sizeof(T)));
        }

        __host__ virtual ~HostDevPtr()
        {
            if (m_hostPtr) { cudaFreeHost(m_hostPtr); }
            if (m_devPtr) { cudaFree(m_devPtr); }
        }

        __host__ __device__ HostDevPtr(const HostDevPtr&) = delete;

        __host__ __device__ HostDevPtr& operator=(const HostDevPtr&) = delete;

        __host__ HostDevPtr(HostDevPtr&& other) noexcept
                : m_hostPtr(other.m_hostPtr),
                  m_devPtr(other.m_devPtr),
                  m_size(other.m_size)
        {
            other.m_hostPtr = nullptr;
            other.m_devPtr = nullptr;
            other.m_size = 0;
        }

        __host__ HostDevPtr& operator=(HostDevPtr&& other) noexcept
        {
            if (this != &other) {
                if (m_hostPtr) { cudaFreeHost(m_hostPtr); }
                if (m_devPtr) { cudaFree(m_devPtr); }

                m_hostPtr = other.m_hostPtr;
                m_devPtr = other.m_devPtr;
                m_size = other.m_size;

                other.m_hostPtr = nullptr;
                other.m_devPtr = nullptr;
                other.m_size = 0;
            }

            return *this;
        }

        __host__ void resizeAndDiscardData(unsigned int newSize)
        {
            if (m_hostPtr) { CUDA_CHECK(cudaFreeHost(m_hostPtr)); }
            if (m_devPtr) { CUDA_CHECK(cudaFree(m_devPtr)); }

            CUDA_CHECK(cudaMallocHost(&m_hostPtr, newSize * sizeof(T)));
            CUDA_CHECK(cudaMalloc(&m_devPtr, newSize * sizeof(T)));
            m_size = newSize;
        }

        __host__ void resizeAndCopy(unsigned int newSize)
        {
            T* oldHostPtr = m_hostPtr;
            T* oldDevPtr = m_devPtr;
            unsigned int oldSize = m_size;

            CUDA_CHECK(cudaMallocHost(&m_hostPtr, newSize * sizeof(T)));
            CUDA_CHECK(cudaMalloc(&m_devPtr, newSize * sizeof(T)));
            m_size = newSize;

            unsigned int sizeToCopy = std::min(oldSize, newSize);

            if (oldHostPtr) {
                CUDA_CHECK(cudaMemcpy(m_hostPtr, oldHostPtr, sizeToCopy * sizeof(T), cudaMemcpyHostToHost));
                CUDA_CHECK(cudaFreeHost(oldHostPtr));
            }

            if (oldDevPtr) {
                CUDA_CHECK(cudaMemcpy(m_devPtr, oldDevPtr, sizeToCopy * sizeof(T), cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaFree(oldDevPtr));
            }
        }

        __host__ void loadToDev() const
        {
            CUDA_CHECK(cudaMemcpy(m_devPtr, m_hostPtr, m_size * sizeof(T), cudaMemcpyHostToDevice));
        }

        __host__ void loadToHost() const
        {
            CUDA_CHECK(cudaMemcpy(m_hostPtr, m_devPtr, m_size * sizeof(T), cudaMemcpyDeviceToHost));
        }

        [[nodiscard]] __host__ T* hostPtr() const
        { return m_hostPtr; }

        [[nodiscard]] __host__ T* devPtr() const
        { return m_devPtr; }

        [[nodiscard]] __host__ unsigned int size() const
        { return m_size; }

    private:
        T* m_hostPtr;
        T* m_devPtr;
        unsigned int m_size;

    }; // HostDevPtr

    template<typename T>
    class HostDevDynamicArray : private HostDevPtr<T>
    {
    public:
        __host__ HostDevDynamicArray() : m_afterLastIdx(0), HostDevPtr<T>(DARR_BASE_SIZE)
        {
        }

        void pushBack(const T& t)
        {
            if (m_afterLastIdx == this->size()) { this->resizeAndCopy(this->size() * 2); }

            (*this)[m_afterLastIdx] = t;
        }

        using HostDevPtr<T>::loadToDev;
        using HostDevPtr<T>::loadToHost;
        using HostDevPtr<T>::hostPtr;
        using HostDevPtr<T>::devPtr;
        using HostDevPtr<T>::size;

    private:
        unsigned int m_afterLastIdx;

    }; // HostDevDynamicArray

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

        [[nodiscard]] __host__ __device__ Quaternion<T> operator*(const T& scalar) const
        {
            return {scalar * m_realPart, Vector3<T>(scalar * m_imaginaryVec)};
        }

        [[nodiscard]] __host__ __device__ friend Quaternion<T>
        operator*(const T& scalar, const Quaternion<T>& quaternion)
        {
            return quaternion * scalar;
        }

        [[nodiscard]] __host__ __device__ Quaternion<T> operator+(const Quaternion& other) const
        {
            return {m_realPart + other.m_realPart, m_imaginaryVec + other.m_imaginaryVec};
        }

        [[nodiscard]] __host__ __device__ Quaternion<T> operator*(const Quaternion& other) const
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

        [[nodiscard]] __host__ __device__ T realPart() const
        { return m_realPart; }

        [[nodiscard]] __host__ __device__ Vector3<T> imaginaryVec() const
        { return m_imaginaryVec; }

    private:
        T m_realPart;
        Vector3<T> m_imaginaryVec;

    }; // Quaternion

    /* Structs */
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

    struct fragment_t
    {
        bool occupied;
        Vector3<float> coords;
        Vector3<float> normal;

        __host__ __device__ explicit fragment_t(
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

} // Custosh


#endif // CUSTOSH_UTILITY_CUH
