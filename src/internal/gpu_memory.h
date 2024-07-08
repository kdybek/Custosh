#ifndef CUSTOSH_GPU_MEMORY_H
#define CUSTOSH_GPU_MEMORY_H


#include <cuda_runtime.h>

#define CUSTOSH_CUDA_CHECK(call) \
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

namespace custosh
{
    template<typename T>
    class HostPtr
    {
    public:
        explicit HostPtr(unsigned int size) : m_hostPtr(nullptr), m_size(size)
        {
            CUSTOSH_CUDA_CHECK(cudaMallocHost(&m_hostPtr, size * sizeof(T)));
        }

        virtual ~HostPtr()
        {
            if (m_hostPtr) { cudaFreeHost(m_hostPtr); }
        }

        HostPtr(const HostPtr&) = delete;

        HostPtr& operator=(const HostPtr&) = delete;

        HostPtr(HostPtr&& other) noexcept: m_hostPtr(other.m_hostPtr), m_size(other.m_size)
        {
            other.m_hostPtr = nullptr;
            other.m_size = 0;
        }

        HostPtr& operator=(HostPtr&& other) noexcept
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

        void resizeAndDiscardData(unsigned int newSize)
        {
            if (m_size == newSize) { return; }

            if (m_hostPtr) { CUSTOSH_CUDA_CHECK(cudaFreeHost(m_hostPtr)); }

            CUSTOSH_CUDA_CHECK(cudaMallocHost(&m_hostPtr, newSize * sizeof(T)));
            m_size = newSize;
        }

        void resizeAndCopy(unsigned int newSize)
        {
            if (m_size == newSize) { return; }

            T* oldHostPtr = m_hostPtr;
            unsigned int oldSize = m_size;

            CUSTOSH_CUDA_CHECK(cudaMallocHost(&m_hostPtr, newSize * sizeof(T)));
            m_size = newSize;

            unsigned int sizeToCopy = std::min(oldSize, newSize);

            if (oldHostPtr) {
                CUSTOSH_CUDA_CHECK(cudaMemcpy(m_hostPtr, oldHostPtr, sizeToCopy * sizeof(T), cudaMemcpyHostToHost));
                CUSTOSH_CUDA_CHECK(cudaFreeHost(oldHostPtr));
            }
        }

        void loadToDev(T* devPtr) const
        {
            CUSTOSH_CUDA_CHECK(cudaMemcpy(devPtr, m_hostPtr, m_size * sizeof(T), cudaMemcpyHostToDevice));
        }

        [[nodiscard]] inline T* get() const
        { return m_hostPtr; }

        [[nodiscard]] inline unsigned int size() const
        { return m_size; }

    private:
        T* m_hostPtr;
        unsigned int m_size;

    }; // HostPtr

    template<typename T>
    class DevPtr
    {
    public:
        explicit DevPtr(unsigned int size) : m_devPtr(nullptr), m_size(size)
        {
            CUSTOSH_CUDA_CHECK(cudaMalloc(&m_devPtr, size * sizeof(T)));
        }

        virtual ~DevPtr()
        {
            if (m_devPtr) { cudaFree(m_devPtr); }
        }

        DevPtr(const DevPtr&) = delete;

        DevPtr& operator=(const DevPtr&) = delete;

        DevPtr(DevPtr&& other) noexcept: m_devPtr(other.m_devPtr), m_size(other.m_size)
        {
            other.m_devPtr = nullptr;
            other.m_size = 0;
        }

        DevPtr& operator=(DevPtr&& other) noexcept
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

        void resizeAndDiscardData(unsigned int newSize)
        {
            if (m_size == newSize) { return; }

            if (m_devPtr) { CUSTOSH_CUDA_CHECK(cudaFree(m_devPtr)); }

            CUSTOSH_CUDA_CHECK(cudaMalloc(&m_devPtr, newSize * sizeof(T)));
            m_size = newSize;
        }

        void resizeAndCopy(unsigned int newSize)
        {
            if (m_size == newSize) { return; }

            T* oldDevPtr = m_devPtr;
            unsigned int oldSize = m_size;

            CUSTOSH_CUDA_CHECK(cudaMalloc(&m_devPtr, newSize * sizeof(T)));
            m_size = newSize;

            unsigned int sizeToCopy = std::min(oldSize, newSize);

            if (oldDevPtr) {
                CUSTOSH_CUDA_CHECK(cudaMemcpy(m_devPtr, oldDevPtr, sizeToCopy * sizeof(T), cudaMemcpyDeviceToDevice));
                CUSTOSH_CUDA_CHECK(cudaFree(oldDevPtr));
            }
        }

        void loadToHost(T* hostPtr) const
        {
            CUSTOSH_CUDA_CHECK(cudaMemcpy(hostPtr, m_devPtr, m_size * sizeof(T), cudaMemcpyDeviceToHost));
        }

        [[nodiscard]] inline T* get() const
        { return m_devPtr; }

        [[nodiscard]] inline unsigned int size() const
        { return m_size; }

    private:
        T* m_devPtr;
        unsigned int m_size;

    }; // DevPtr

} // custosh


#endif // CUSTOSH_GPU_MEMORY_H
