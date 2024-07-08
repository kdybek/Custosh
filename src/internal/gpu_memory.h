#ifndef CUSTOSH_GPU_MEMORY_H
#define CUSTOSH_GPU_MEMORY_H


#include <cuda_runtime.h>

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
} while (0)

#define HOST_MEMBER __host__ constexpr
#define HOST_GETTER __host__ inline constexpr

namespace Custosh
{
    template<typename T>
    class HostPtr
    {
    public:
        HOST_MEMBER explicit HostPtr(unsigned int size) : m_hostPtr(nullptr), m_size(size)
        {
            CUDA_CHECK(cudaMallocHost(&m_hostPtr, size * sizeof(T)));
        }

        HOST_MEMBER virtual ~HostPtr()
        {
            if (m_hostPtr) { cudaFreeHost(m_hostPtr); }
        }

        __host__ __device__ HostPtr(const HostPtr&) = delete;

        __host__ __device__ HostPtr& operator=(const HostPtr&) = delete;

        HOST_MEMBER HostPtr(HostPtr&& other) noexcept: m_hostPtr(other.m_hostPtr), m_size(other.m_size)
        {
            other.m_hostPtr = nullptr;
            other.m_size = 0;
        }

        HOST_MEMBER HostPtr& operator=(HostPtr&& other) noexcept
        {
            if (this != &other) {
                if (m_hostPtr) { cudaFreeHost(m_hostPtr); }

                m_hostPtr = other.m_hostPtr;
                m_size = other.m_size;

                other.m_hostPtr = nullptr;
                other.m_size = 0;
            }

            return *this;
        }

        HOST_MEMBER void resizeAndDiscardData(unsigned int newSize)
        {
            if (m_size == newSize) { return; }

            if (m_hostPtr) { CUDA_CHECK(cudaFreeHost(m_hostPtr)); }

            CUDA_CHECK(cudaMallocHost(&m_hostPtr, newSize * sizeof(T)));
            m_size = newSize;
        }

        HOST_MEMBER void resizeAndCopy(unsigned int newSize)
        {
            if (m_size == newSize) { return; }

            T* oldHostPtr = m_hostPtr;
            unsigned int oldSize = m_size;

            CUDA_CHECK(cudaMallocHost(&m_hostPtr, newSize * sizeof(T)));
            m_size = newSize;

            unsigned int sizeToCopy = std::min(oldSize, newSize);

            if (oldHostPtr) {
                CUDA_CHECK(cudaMemcpy(m_hostPtr, oldHostPtr, sizeToCopy * sizeof(T), cudaMemcpyHostToHost));
                CUDA_CHECK(cudaFreeHost(oldHostPtr));
            }
        }

        HOST_MEMBER void loadToDev(T* devPtr) const
        {
            CUDA_CHECK(cudaMemcpy(devPtr, m_hostPtr, m_size * sizeof(T), cudaMemcpyHostToDevice));
        }

        [[nodiscard]] HOST_GETTER T* get() const
        { return m_hostPtr; }

        [[nodiscard]] HOST_GETTER unsigned int size() const
        { return m_size; }

    private:
        T* m_hostPtr;
        unsigned int m_size;

    }; // HostPtr

    template<typename T>
    class DevPtr
    {
    public:
        HOST_MEMBER explicit DevPtr(unsigned int size) : m_devPtr(nullptr), m_size(size)
        {
            CUDA_CHECK(cudaMalloc(&m_devPtr, size * sizeof(T)));
        }

        HOST_MEMBER virtual ~DevPtr()
        {
            if (m_devPtr) { cudaFree(m_devPtr); }
        }

        __host__ __device__ DevPtr(const DevPtr&) = delete;

        __host__ __device__ DevPtr& operator=(const DevPtr&) = delete;

        HOST_MEMBER DevPtr(DevPtr&& other) noexcept: m_devPtr(other.m_devPtr), m_size(other.m_size)
        {
            other.m_devPtr = nullptr;
            other.m_size = 0;
        }

        HOST_MEMBER DevPtr& operator=(DevPtr&& other) noexcept
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

        HOST_MEMBER void resizeAndDiscardData(unsigned int newSize)
        {
            if (m_size == newSize) { return; }

            if (m_devPtr) { CUDA_CHECK(cudaFree(m_devPtr)); }

            CUDA_CHECK(cudaMalloc(&m_devPtr, newSize * sizeof(T)));
            m_size = newSize;
        }

        HOST_MEMBER void resizeAndCopy(unsigned int newSize)
        {
            if (m_size == newSize) { return; }

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

        HOST_MEMBER void loadToHost(T* hostPtr) const
        {
            CUDA_CHECK(cudaMemcpy(hostPtr, m_devPtr, m_size * sizeof(T), cudaMemcpyDeviceToHost));
        }

        [[nodiscard]] HOST_GETTER T* get() const
        { return m_devPtr; }

        [[nodiscard]] HOST_GETTER unsigned int size() const
        { return m_size; }

    private:
        T* m_devPtr;
        unsigned int m_size;

    }; // DevPtr

} // Custosh


#endif // CUSTOSH_GPU_MEMORY_H
