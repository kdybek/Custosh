#include "Renderer.cuh"

#include <iostream>

#include "WindowsConsoleScreenBuffer.h"

#define BASE_DEV_WSPACE_SIZE 1000

namespace Custosh::Renderer
{
    namespace
    {
        /* Device global variables */
        __constant__ const char* g_devASCIIByBrightness =
                R"( .'`,_^"-+:;!><~?iI[]{}1()|\/tfjrnxuvczXYUJCLQ0OZmwqpdkbhao*#MW&8%B@$)";
        __constant__ const unsigned int g_devNumASCII = 69; // TODO: make sure it's right

        /* Host global variables */
        unsigned int g_hostScreenRows = 70;
        unsigned int g_hostScreenCols = 70;
        WindowsConsoleScreenBuffer g_hostActiveBuf;
        WindowsConsoleScreenBuffer g_hostInactiveBuf;

        /* Device working space pointers */
        HostDevPtr<Vector2<float>> g_hostVertex2DDevPtr(BASE_DEV_WSPACE_SIZE);
        HostDevPtr<boundingBox_t> g_hostBoundingBoxDevPtr(BASE_DEV_WSPACE_SIZE);
        HostDevPtr<float> g_hostTriangleCross2DDevPtr(BASE_DEV_WSPACE_SIZE);
        HostDevPtr<Vector3<float>> g_hostTriangleNormalDevPtr(BASE_DEV_WSPACE_SIZE);
        HostDevPtr<fragment_t> g_hostFragmentDevPtr(g_hostScreenRows * g_hostScreenCols);
        HostDevPtr<char> g_hostCharPtr(g_hostScreenRows * g_hostScreenCols);

        /* Device auxiliary functions */
        template<typename T>
        [[nodiscard]] __device__ T max3(T a, T b, T c)
        {
            return max(max(a, b), c);
        }

        template<typename T>
        [[nodiscard]] __device__ T min3(T a, T b, T c)
        {
            return min(min(a, b), c);
        }

        template<typename T>
        __device__ void swap(T& a, T& b)
        {
            T temp = a;
            a = b;
            b = temp;
        }

        [[nodiscard]] __device__ char brightnessToASCII(float brightness)
        {
            unsigned int idx = ceil(brightness * static_cast<float>(g_devNumASCII - 1));
            return g_devASCIIByBrightness[idx];
        }

        [[nodiscard]] __device__ boundingBox_t findBounds(const triangle2D_t& triangle2D,
                                                          unsigned int rows,
                                                          unsigned int cols)
        {
            boundingBox_t boundingBox;
            float xMax = max3(ceil(triangle2D.p0.x()),
                              ceil(triangle2D.p1.x()),
                              ceil(triangle2D.p2.x()));
            float xMin = min3(floor(triangle2D.p0.x()),
                              floor(triangle2D.p1.x()),
                              floor(triangle2D.p2.x()));
            float yMax = max3(ceil(triangle2D.p0.y()),
                              ceil(triangle2D.p1.y()),
                              ceil(triangle2D.p2.y()));
            float yMin = min3(floor(triangle2D.p0.y()),
                              floor(triangle2D.p1.y()),
                              floor(triangle2D.p2.y()));

            boundingBox.xMax = min(static_cast<int>(xMax), static_cast<int>(rows - 1));
            boundingBox.xMin = max(static_cast<int>(xMin), 0);
            boundingBox.yMax = min(static_cast<int>(yMax), static_cast<int>(cols - 1));
            boundingBox.yMin = max(static_cast<int>(yMin), 0);
            return boundingBox;
        }

        [[nodiscard]] __device__ bool isBottomOrRight(const Vector2<float>& a,
                                                      const Vector2<float>& b)
        {
            Vector2<float> edge = {b.x() - a.x(), b.y() - a.y()};
            bool bottomEdge = edge.y() == 0 && edge.x() < 0;
            bool rightEdge = edge.y() < 0;

            return bottomEdge || rightEdge;
        }

        [[nodiscard]] __device__ float cross2D(const Vector2<float>& a,
                                               const Vector2<float>& b,
                                               const Vector2<float>& c)
        {
            Vector2<float> ab = {b.x() - a.x(), b.y() - a.y()};
            Vector2<float> ac = {c.x() - a.x(), c.y() - a.y()};
            return ab.x() * ac.y() - ab.y() * ac.x();
        }

        [[nodiscard]] __device__ bool inBoundingBox(const boundingBox_t& boundingBox,
                                                    const Vector2<float>& p)
        {
            if (p.x() >= static_cast<float>(boundingBox.xMin) &&
                p.x() <= static_cast<float>(boundingBox.xMax) &&
                p.y() >= static_cast<float>(boundingBox.yMin) &&
                p.y() <= static_cast<float>(boundingBox.yMax)) {
                return true;
            }
            else { return false; }
        }

        [[nodiscard]] __device__ bool inTriangle(const triangle2D_t& triangle2D,
                                                 const boundingBox_t& boundingBox,
                                                 const Vector2<float>& p,
                                                 float triangleArea2x,
                                                 barycentricCoords_t& barycentricCoords)
        {
            if (!inBoundingBox(boundingBox, p)) { return false; }

            float w0 = cross2D(triangle2D.p1, p, triangle2D.p2);
            float w1 = cross2D(triangle2D.p2, p, triangle2D.p0);
            float w2 = cross2D(triangle2D.p0, p, triangle2D.p1);

            if (w0 == 0 && isBottomOrRight(triangle2D.p1, triangle2D.p2)) { return false; }
            if (w1 == 0 && isBottomOrRight(triangle2D.p2, triangle2D.p0)) { return false; }
            if (w2 == 0 && isBottomOrRight(triangle2D.p0, triangle2D.p1)) { return false; }

            barycentricCoords.alpha = w0 / triangleArea2x;
            barycentricCoords.beta = w1 / triangleArea2x;
            barycentricCoords.gamma = w2 / triangleArea2x;

            return (w0 >= 0.f && w1 >= 0.f && w2 >= 0.f);
        }

        [[nodiscard]] __device__ Vector2<float> applyPerspective(const Vector3<float>& p,
                                                                 const PerspectiveProjMatrix& ppm)
        {
            Vector4<float> pPerspective = Vector4<float>(ppm * p.toHomogeneous()).normalizeW();
            return {pPerspective.x(), pPerspective.y()};
        }

        [[nodiscard]] __device__ Vector3<float>
        getCartesianCoords(const triangle3D_t& triangle3D,
                           const barycentricCoords_t& bc)
        {
            return {triangle3D.p0.x() * bc.alpha + triangle3D.p1.x() * bc.beta + triangle3D.p2.x() * bc.gamma,
                    triangle3D.p0.y() * bc.alpha + triangle3D.p1.y() * bc.beta + triangle3D.p2.y() * bc.gamma,
                    triangle3D.p0.z() * bc.alpha + triangle3D.p1.z() * bc.beta + triangle3D.p2.z() * bc.gamma};
        }

        [[nodiscard]] __device__ float distanceSq(const Vector3<float>& a,
                                                  const Vector3<float>& b)
        {
            return static_cast<float>(pow((a.x() - b.x()), 2) + pow((a.y() - b.y()), 2) + pow((a.z() - b.z()), 2));
        }

        [[nodiscard]] __device__ float cosine3D(const Vector3<float>& center,
                                                const Vector3<float>& p1,
                                                const Vector3<float>& p2)
        {
            auto vec1 = Vector3<float>(p1 - center);
            auto vec2 = Vector3<float>(p2 - center);
            float dist1 = std::sqrt(distanceSq(center, p1));
            float dist2 = std::sqrt(distanceSq(center, p2));

            return vec1.dot(vec2) / (dist1 * dist2);
        }

        [[nodiscard]] __device__ float pointBrightness(const fragment_t& p,
                                                       const lightSource_t& ls)
        {
            float distSq = distanceSq(p.coords, ls.coords);
            float cos = cosine3D(p.coords, Vector3<float>(p.coords + p.normal), ls.coords);

            return clamp(max(cos, 0.f) * ls.intensity / distSq, 0.f, 1.f);
        }

        // The vertices are clockwise oriented, but we're looking from 0 towards positive z values.
        [[nodiscard]] __device__ Vector3<float> triangleNormal(const triangle3D_t& triangle3D)
        {
            Vector3<float> res = Vector3<float>(triangle3D.p1 - triangle3D.p0).cross(
                    Vector3<float>(triangle3D.p2 - triangle3D.p0));

            return Vector3<float>(res.normalized());
        }

        [[nodiscard]] __device__ triangle2D_t getTriangle2D(const triangleIndices_t& triangleIndices,
                                                            const Vector2<float>* vertex2DPtr)
        {
            return triangle2D_t(vertex2DPtr[triangleIndices.p0],
                                vertex2DPtr[triangleIndices.p1],
                                vertex2DPtr[triangleIndices.p2]);
        }

        [[nodiscard]] __device__ triangle3D_t getTriangle3D(const triangleIndices_t& triangleIndices,
                                                            const Vector3<float>* vertex3DPtr)
        {
            return triangle3D_t(vertex3DPtr[triangleIndices.p0],
                                vertex3DPtr[triangleIndices.p1],
                                vertex3DPtr[triangleIndices.p2]);
        }

        /* Host auxiliary functions */
        __host__ void swapBuffers()
        {
            g_hostInactiveBuf.activate();
            std::swap(g_hostActiveBuf, g_hostInactiveBuf);
        }

        /* Kernels */
        // TODO: vertex manipulation (translation, rotation, etc.)
        __global__ void vertexShader(const Vector3<float>* vertex3DPtr,
                                     unsigned int numVertices,
                                     PerspectiveProjMatrix ppm,
                                     Vector2<float>* vertex2DPtr)
        {
            const unsigned int x = threadIdx.x;

            if (x >= numVertices) { return; }

            vertex2DPtr[x] = applyPerspective(vertex3DPtr[x], ppm);
        }

        __global__ void populateTriangleParams(unsigned int rows,
                                               unsigned int cols,
                                               triangleIndices_t* indexPtr,
                                               unsigned int numTriangles,
                                               const Vector2<float>* vertex2DPtr,
                                               const Vector3<float>* vertex3DPtr,
                                               float* cross2DPtr,
                                               Vector3<float>* normalPtr,
                                               boundingBox_t* boundingBoxPtr)
        {
            const unsigned int x = threadIdx.x;

            if (x >= numTriangles) { return; }

            triangle2D_t triangle2D = getTriangle2D(indexPtr[x], vertex2DPtr);
            triangle3D_t triangle3D = getTriangle3D(indexPtr[x], vertex3DPtr);

            float cross = cross2D(triangle2D.p0, triangle2D.p2, triangle2D.p1);

            // In other functions the triangles' vertices are assumed to be in a clockwise order.
            if (cross < 0.f) {
                swap(triangle2D.p0, triangle2D.p1);
                swap(triangle3D.p0, triangle3D.p1);
                swap(indexPtr[x].p0, indexPtr[x].p1);
                cross *= -1;
            }

            cross2DPtr[x] = cross;
            normalPtr[x] = triangleNormal(triangle3D);
            boundingBoxPtr[x] = findBounds(triangle2D, rows, cols);
        }

        __global__ void fragmentShader1(unsigned int rows,
                                        unsigned int cols,
                                        const triangleIndices_t* indexPtr,
                                        unsigned int numTriangles,
                                        const Vector2<float>* vertex2DPtr,
                                        const Vector3<float>* vertex3DPtr,
                                        const float* cross2DPtr,
                                        const Vector3<float>* normalPtr,
                                        const boundingBox_t* boundingBoxPtr,
                                        fragment_t* fragmentPtr)
        {
            const unsigned int x = threadIdx.x;
            const unsigned int y = threadIdx.y;

            if (x >= rows || y >= cols) { return; }

            fragmentPtr[x * cols + y].occupied = false;

            for (unsigned int k = 0; k < numTriangles; ++k) {
                triangle2D_t triangle2D = getTriangle2D(indexPtr[k], vertex2DPtr);
                triangle3D_t triangle3D = getTriangle3D(indexPtr[k], vertex3DPtr);
                float triangleArea2x = cross2DPtr[k];
                Vector3<float> normal = normalPtr[k];
                boundingBox_t boundingBox = boundingBoxPtr[k];
                barycentricCoords_t bc;

                if (triangleArea2x == 0.f) { continue; }

                if (inTriangle(triangle2D,
                               boundingBox,
                               Vector2<float>({static_cast<float>(x), static_cast<float>(y)}),
                               triangleArea2x,
                               bc)) {
                    Vector3<float> projectedPoint = getCartesianCoords(triangle3D, bc);
                    fragment_t& screenPoint = fragmentPtr[x * cols + y];

                    if (!screenPoint.occupied || screenPoint.coords.z() > projectedPoint.z()) {
                        screenPoint.occupied = true;
                        screenPoint.coords = projectedPoint;
                        screenPoint.normal = normal;
                    }
                }
            }
        }

        __global__ void fragmentShader2(const fragment_t* fragmentPtr,
                                        unsigned int rows,
                                        unsigned int cols,
                                        lightSource_t ls,
                                        char* characters)
        {
            const unsigned int x = threadIdx.x;
            const unsigned int y = threadIdx.y;

            if (x >= rows || y >= cols) { return; }

            const fragment_t& screenPoint = fragmentPtr[x * cols + y];

            if (screenPoint.occupied) {
                characters[x * cols + y] = brightnessToASCII(pointBrightness(screenPoint, ls));
            }
            else { characters[x * cols + y] = brightnessToASCII(0.f); }
        }

    } // anonymous

    __host__ void drawMesh(const Mesh& mesh, const PerspectiveProjMatrix& ppm, const lightSource_t& ls)
    {
        // TODO: working space resizing!!!

        mesh.hostDevVerticesPtr().loadToDev();
        mesh.hostDevIndicesPtr().loadToDev();

        const Vector3<float>* vertex3DPtr = mesh.hostDevVerticesPtr().devPtr();
        triangleIndices_t* indexPtr = mesh.hostDevIndicesPtr().devPtr();

        unsigned int maxThreadsPerBlock = 1024;

        unsigned int numVertices = mesh.hostDevVerticesPtr().size();
        unsigned int numTriangles = mesh.hostDevIndicesPtr().size();

        unsigned int threadsPerBlockVShader = std::min(numVertices, maxThreadsPerBlock);
        unsigned int numBlocksVShader = (numVertices + threadsPerBlockVShader - 1) / threadsPerBlockVShader;

        vertexShader<<<numBlocksVShader, threadsPerBlockVShader>>>(vertex3DPtr,
                                                                   numVertices,
                                                                   ppm,
                                                                   g_hostVertex2DDevPtr.devPtr());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        unsigned int threadsPerBlockTParam = std::min(numTriangles, maxThreadsPerBlock);
        unsigned int numBlocksTParam = (numTriangles + threadsPerBlockTParam - 1) / threadsPerBlockTParam;

        populateTriangleParams<<<numBlocksTParam, threadsPerBlockTParam>>>(g_hostScreenRows,
                                                                           g_hostScreenCols,
                                                                           indexPtr,
                                                                           numTriangles,
                                                                           g_hostVertex2DDevPtr.devPtr(),
                                                                           vertex3DPtr,
                                                                           g_hostTriangleCross2DDevPtr.devPtr(),
                                                                           g_hostTriangleNormalDevPtr.devPtr(),
                                                                           g_hostBoundingBoxDevPtr.devPtr());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        unsigned int threadsPerBlockFShader = std::min(g_hostScreenRows * g_hostScreenCols, maxThreadsPerBlock);
        unsigned int numBlocksFShader = (g_hostScreenRows * g_hostScreenCols + threadsPerBlockFShader - 1) / threadsPerBlockFShader;

        fragmentShader1<<<numBlocksFShader, threadsPerBlockFShader>>>(g_hostScreenRows,
                                                                      g_hostScreenCols,
                                                                      indexPtr,
                                                                      numTriangles,
                                                                      g_hostVertex2DDevPtr.devPtr(),
                                                                      vertex3DPtr,
                                                                      g_hostTriangleCross2DDevPtr.devPtr(),
                                                                      g_hostTriangleNormalDevPtr.devPtr(),
                                                                      g_hostBoundingBoxDevPtr.devPtr(),
                                                                      g_hostFragmentDevPtr.devPtr());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        g_hostFragmentDevPtr.loadToHost();
        std::cout << g_hostFragmentDevPtr.hostPtr()[35 * 70 + 35].occupied << '\n';
        std::cout << g_hostFragmentDevPtr.hostPtr()[35 * 70 + 35].coords.x() << '\n';
        std::cout << g_hostFragmentDevPtr.hostPtr()[35 * 70 + 35].coords.y() << '\n';
        std::cout << g_hostFragmentDevPtr.hostPtr()[35 * 70 + 35].coords.z() << '\n';

        return;

        fragmentShader2<<<numBlocksFShader, threadsPerBlockFShader>>>(g_hostFragmentDevPtr.devPtr(),
                                                                      g_hostScreenRows,
                                                                      g_hostScreenCols,
                                                                      ls,
                                                                      g_hostCharPtr.devPtr());
        CUDA_CHECK(cudaGetLastError());
        g_hostCharPtr.loadToHost();
        g_hostInactiveBuf.draw(g_hostCharPtr.hostPtr(), g_hostScreenRows, g_hostScreenCols);
        swapBuffers();
    }

} // Custosh::Rendere