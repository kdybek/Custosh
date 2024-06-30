#ifndef CUSTOSH_TRIANGLEREGISTRY_CUH
#define CUSTOSH_TRIANGLEREGISTRY_CUH


#include <cuda_runtime.h>

#include "Utility.cuh"
#include "Mesh.h"

namespace Custosh
{

    class TriangleRegistry
    {
    public:
        __host__ TriangleRegistry() : m_hostTrianglesVec(),
                                      m_devTrianglesPtr(nullptr),
                                      m_devTrianglesSize(0)
        {
        }

        __host__ ~TriangleRegistry()
        {
            cudaFree(m_devTrianglesPtr);
        }

        __host__ __device__ TriangleRegistry(const TriangleRegistry&) = delete;

        __host__ __device__ TriangleRegistry& operator=(const TriangleRegistry&) = delete;

        __host__ void add(const triangle3D_t& triangle)
        {
            m_hostTrianglesVec.push_back(triangle);
        }

        __host__ void add(const Mesh& mesh)
        {
            for (const triangle3D_t& triangle: mesh.getTriangles()) {
                add(triangle);
            }
        }

        __host__ void loadToDev()
        {
            CUDA_CHECK(cudaFree(m_devTrianglesPtr));

            m_devTrianglesSize = m_hostTrianglesVec.size();

            CUDA_CHECK(cudaMalloc(&m_devTrianglesPtr, sizeof(triangle3D_t) * m_devTrianglesSize));
            CUDA_CHECK(cudaMemcpy(m_devTrianglesPtr,
                                  m_hostTrianglesVec.data(),
                                  m_devTrianglesSize * sizeof(triangle3D_t),
                                  cudaMemcpyHostToDevice));
        }

        __host__ __device__ [[nodiscard]] unsigned int size() const
        {
#ifdef __CUDA_ARCH__
            return m_devTrianglesSize;
#else
            return m_hostTrianglesVec.size();
#endif
        }

        __device__ triangle3D_t getDeviceTriangle(unsigned int idx)
        {
            return m_devTrianglesPtr[idx];
        }

    private:
        std::vector<triangle3D_t> m_hostTrianglesVec;
        triangle3D_t* m_devTrianglesPtr;
        unsigned int m_devTrianglesSize;

    }; // TriangleRegistry

} // Custosh


#endif // CUSTOSH_TRIANGLEREGISTRY_CUH
