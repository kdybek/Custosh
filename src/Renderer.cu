#include "Renderer.cuh"

#include "WindowsConsoleScreenBuffer.h"

#define BASE_DEV_WSPACE_SIZE 8
#define CCV_MIN_CORNER {-1.f, -1.f, -1.f}
#define CCV_MAX_CORNER {1.f, 1.f, 1.f}
#define PM_NEAR_PLANE 1.f
#define PM_FAR_PLANE 10.f
#define MAX_THREADS_PER_BLOCK 256
#define THREADS_PER_BLOCK_X 16
#define THREADS_PER_BLOCK_Y 16

namespace Custosh::Renderer
{
    namespace
    {
        /* Device global variables */
        __constant__ const char* g_devASCIIByBrightness =
                R"( `.-':_,^=;><+!rc*/z?sLTv)J7(|Fi{C}fI31tlu[neoZ5Yxjya]2ESwqkP6h9d4VpOGbUAKXHm8RD#$Bg0MNWQ%&@)";
        __constant__ const unsigned int g_devNumASCII = 94; // TODO: make sure it's right

        /* Host global variables */
        WindowsConsoleScreenBuffer g_hostActiveBuf;
        WindowsConsoleScreenBuffer g_hostInactiveBuf;
        HostPtr<char> g_hostCharHostPtr(BASE_DEV_WSPACE_SIZE);

        /* Device working space pointers */
        DevPtr<Vertex3D> g_hostVertex3DDevPtr(BASE_DEV_WSPACE_SIZE);
        DevPtr<triangleIndices_t> g_hostIndexDevPtr(BASE_DEV_WSPACE_SIZE);
        DevPtr<Vertex2D> g_hostVertex2DDevPtr(BASE_DEV_WSPACE_SIZE);
        DevPtr<boundingBox_t> g_hostBoundingBoxDevPtr(BASE_DEV_WSPACE_SIZE);
        DevPtr<float> g_hostTriangleCross2DDevPtr(BASE_DEV_WSPACE_SIZE);
        DevPtr<Vector3<float>> g_hostTriangleNormalDevPtr(BASE_DEV_WSPACE_SIZE);
        DevPtr<char> g_hostCharDevPtr(BASE_DEV_WSPACE_SIZE);

        /* Device auxiliary functions */
        [[nodiscard]] __device__ char brightnessToASCII(float brightness)
        {
            unsigned int idx = ceil(brightness * static_cast<float>(g_devNumASCII - 1));
            return g_devASCIIByBrightness[idx];
        }

        [[nodiscard]] __device__ boundingBox_t findBounds(const triangle2D_t& triangle2D)
        {
            boundingBox_t boundingBox;

            boundingBox.xMax = max3(ceil(triangle2D.p0.x()),
                                    ceil(triangle2D.p1.x()),
                                    ceil(triangle2D.p2.x()));
            boundingBox.xMin = min3(floor(triangle2D.p0.x()),
                                    floor(triangle2D.p1.x()),
                                    floor(triangle2D.p2.x()));
            boundingBox.yMax = max3(ceil(triangle2D.p0.y()),
                                    ceil(triangle2D.p1.y()),
                                    ceil(triangle2D.p2.y()));
            boundingBox.yMin = min3(floor(triangle2D.p0.y()),
                                    floor(triangle2D.p1.y()),
                                    floor(triangle2D.p2.y()));

            return boundingBox;
        }

        [[nodiscard]] __device__ bool isBottomOrRight(const Vertex2D& a,
                                                      const Vertex2D& b)
        {
            auto edge = Vector2<float>(b - a);
            bool bottomEdge = edge.y() == 0 && edge.x() < 0;
            bool rightEdge = edge.y() < 0;

            return bottomEdge || rightEdge;
        }

        [[nodiscard]] __device__ float cross2D(const Vertex2D& a,
                                               const Vertex2D& b,
                                               const Vertex2D& c)
        {
            auto ab = Vector2<float>(b - a);
            auto ac = Vector2<float>(c - a);

            return ab.x() * ac.y() - ab.y() * ac.x();
        }

        [[nodiscard]] __device__ bool inBoundingBox(const boundingBox_t& boundingBox,
                                                    const Vertex2D& p)
        {
            if (p.x() >= boundingBox.xMin && p.x() <= boundingBox.xMax &&
                p.y() >= boundingBox.yMin && p.y() <= boundingBox.yMax) {
                return true;
            }
            else { return false; }
        }

        [[nodiscard]] __device__ bool inTriangle(const triangle2D_t& triangle2D,
                                                 const boundingBox_t& boundingBox,
                                                 const Vertex2D& p,
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

        [[nodiscard]] __device__ Vertex2D applyPerspective(const Vertex3D& p,
                                                           const PerspectiveProjMatrix& ppm)
        {
            Vector4<float> pPerspective = Vector4<float>(ppm * p.toHomogeneous()).normalizeW();
            return {pPerspective.x(), pPerspective.y()};
        }

        [[nodiscard]] __device__ Vertex3D getCartesianCoords(const triangle3D_t& triangle3D,
                                                             const barycentricCoords_t& bc)
        {
            return {triangle3D.p0.x() * bc.alpha + triangle3D.p1.x() * bc.beta + triangle3D.p2.x() * bc.gamma,
                    triangle3D.p0.y() * bc.alpha + triangle3D.p1.y() * bc.beta + triangle3D.p2.y() * bc.gamma,
                    triangle3D.p0.z() * bc.alpha + triangle3D.p1.z() * bc.beta + triangle3D.p2.z() * bc.gamma};
        }

        [[nodiscard]] __device__ float distanceSq(const Vertex3D& a, const Vertex3D& b)
        {
            return static_cast<float>(pow((a.x() - b.x()), 2) + pow((a.y() - b.y()), 2) + pow((a.z() - b.z()), 2));
        }

        [[nodiscard]] __device__ float pointBrightness(const fragment_t& p, const lightSource_t& ls)
        {
            float distSq = distanceSq(p.coords, ls.coords);
            auto pointToLightSourceVec = Vector3<float>(ls.coords - p.coords);
            auto pointToLightSourceVecNorm = Vector3<float>(pointToLightSourceVec.normalized());
            float cos = pointToLightSourceVecNorm.dot(p.normal);

            return clamp(max(cos, 0.f) * ls.intensity / distSq, 0.f, 1.f);
        }

        // The vertices are clockwise oriented, but we're looking from 0 towards positive z values.
        [[nodiscard]] __device__ Vector3<float> triangleNormal(const triangle3D_t& triangle3D)
        {
            Vector3<float> normal = Vector3<float>(triangle3D.p1 - triangle3D.p0).cross(
                    Vector3<float>(triangle3D.p2 - triangle3D.p0));

            return Vector3<float>(normal.normalized());
        }

        [[nodiscard]] __device__ triangle2D_t getTriangle2D(const triangleIndices_t& triangleIndices,
                                                            const Vertex2D* vertex2DPtr)
        {
            return triangle2D_t(vertex2DPtr[triangleIndices.p0],
                                vertex2DPtr[triangleIndices.p1],
                                vertex2DPtr[triangleIndices.p2]);
        }

        [[nodiscard]] __device__ triangle3D_t getTriangle3D(const triangleIndices_t& triangleIndices,
                                                            const Vertex3D* vertex3DPtr)
        {
            return triangle3D_t(vertex3DPtr[triangleIndices.p0],
                                vertex3DPtr[triangleIndices.p1],
                                vertex3DPtr[triangleIndices.p2]);
        }

        /* Kernels */
        // TODO: vertex manipulation (translation, rotation, etc.)
        __global__ void vertexShader(const Vertex3D* vertex3DPtr,
                                     unsigned int numVertices,
                                     PerspectiveProjMatrix ppm,
                                     Vertex2D* vertex2DPtr)
        {
            const unsigned int i = threadIdx.x;

            if (i >= numVertices) { return; }

            vertex2DPtr[i] = applyPerspective(vertex3DPtr[i], ppm);
        }

        __global__ void geometryShader(triangleIndices_t* indexPtr,
                                       unsigned int numTriangles,
                                       const Vertex2D* vertex2DPtr,
                                       const Vertex3D* vertex3DPtr,
                                       float* cross2DPtr,
                                       Vector3<float>* normalPtr,
                                       boundingBox_t* boundingBoxPtr)
        {
            const unsigned int i = threadIdx.x;

            if (i >= numTriangles) { return; }

            triangle2D_t triangle2D = getTriangle2D(indexPtr[i], vertex2DPtr);
            triangle3D_t triangle3D = getTriangle3D(indexPtr[i], vertex3DPtr);

            float cross = cross2D(triangle2D.p0, triangle2D.p2, triangle2D.p1);

            // In other functions the triangles' vertices are assumed to be in a clockwise order.
            if (cross < 0.f) {
                swap(triangle2D.p0, triangle2D.p1);
                swap(triangle3D.p0, triangle3D.p1);
                swap(indexPtr[i].p0, indexPtr[i].p1);
                cross *= -1;
            }

            cross2DPtr[i] = cross;
            normalPtr[i] = triangleNormal(triangle3D);
            boundingBoxPtr[i] = findBounds(triangle2D);
        }

        __global__ void fragmentShader(unsigned int rows,
                                       unsigned int cols,
                                       const triangleIndices_t* indexPtr,
                                       unsigned int numTriangles,
                                       const Vertex2D* vertex2DPtr,
                                       const Vertex3D* vertex3DPtr,
                                       const float* cross2DPtr,
                                       const Vector3<float>* normalPtr,
                                       const boundingBox_t* boundingBoxPtr,
                                       lightSource_t ls,
                                       char* characters)
        {
            const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
            const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (y >= rows || x >= cols) { return; }

            fragment_t fragment;

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
                               Vertex2D({static_cast<float>(x), static_cast<float>(y)}),
                               triangleArea2x,
                               bc)) {
                    Vertex3D projectedPoint = getCartesianCoords(triangle3D, bc);

                    if (!fragment.occupied || fragment.coords.z() > projectedPoint.z()) {
                        fragment.occupied = true;
                        fragment.coords = projectedPoint;
                        fragment.normal = normal;
                    }
                }
            }

            if (fragment.occupied) {
                characters[y * cols + x] = brightnessToASCII(pointBrightness(fragment, ls));
            }
            else { characters[y * cols + x] = brightnessToASCII(0.f); }
        }

        /* Host auxiliary functions */
        [[nodiscard]] __host__ PerspectiveProjMatrix CCV2ScreenPPM(unsigned int screenRows, unsigned int screenCols)
        {
            return {PerspectiveMatrix(PM_NEAR_PLANE, PM_FAR_PLANE),
                    OrtProjMatrix(CCV_MIN_CORNER,
                                  CCV_MAX_CORNER,
                                  {0.f, 0.f, 0.f},
                                  {static_cast<float>(screenCols), static_cast<float>(screenRows), 0.f})};
        }

        __host__ void resizeSceneDependentPtrs(unsigned int numVertices, unsigned int numTriangles)
        {
            g_hostVertex3DDevPtr.resizeAndDiscardData(numVertices);
            g_hostVertex2DDevPtr.resizeAndDiscardData(numVertices);

            g_hostIndexDevPtr.resizeAndDiscardData(numTriangles);
            g_hostBoundingBoxDevPtr.resizeAndDiscardData(numTriangles);
            g_hostTriangleCross2DDevPtr.resizeAndDiscardData(numTriangles);
            g_hostTriangleNormalDevPtr.resizeAndDiscardData(numTriangles);
        }

        __host__ void resizeScreenDependentPtrs(unsigned int windowRows, unsigned int windowCols)
        {
            g_hostCharHostPtr.resizeAndDiscardData(windowRows * windowCols);
            g_hostCharDevPtr.resizeAndDiscardData(windowRows * windowCols);
        }

        __host__ void callVertexShader(const PerspectiveProjMatrix& PPM)
        {
            unsigned int numVertices = g_hostVertex3DDevPtr.size();

            unsigned int threadsPerBlock = std::min(numVertices, static_cast<unsigned int>(MAX_THREADS_PER_BLOCK));
            unsigned int numBlocks = (numVertices + threadsPerBlock - 1) / threadsPerBlock;

            vertexShader<<<numBlocks, threadsPerBlock>>>(g_hostVertex3DDevPtr.get(),
                                                         numVertices,
                                                         PPM,
                                                         g_hostVertex2DDevPtr.get());
            CUDA_CHECK(cudaGetLastError());
        }

        __host__ void callGeometryShader()
        {
            unsigned int numTriangles = g_hostIndexDevPtr.size();

            unsigned int threadsPerBlock = std::min(numTriangles, static_cast<unsigned int>(MAX_THREADS_PER_BLOCK));
            unsigned int numBlocks = (numTriangles + threadsPerBlock - 1) / threadsPerBlock;

            geometryShader<<<numBlocks, threadsPerBlock>>>(g_hostIndexDevPtr.get(),
                                                           numTriangles,
                                                           g_hostVertex2DDevPtr.get(),
                                                           g_hostVertex3DDevPtr.get(),
                                                           g_hostTriangleCross2DDevPtr.get(),
                                                           g_hostTriangleNormalDevPtr.get(),
                                                           g_hostBoundingBoxDevPtr.get());
            CUDA_CHECK(cudaGetLastError());
        }

        __host__ void callFragmentShader(const lightSource_t& ls,
                                         unsigned int windowRows,
                                         unsigned int windowCols)
        {
            unsigned int numTriangles = g_hostIndexDevPtr.size();

            dim3 threadsPerBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
            dim3 numBlocks((windowCols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                           (windowRows + threadsPerBlock.y - 1) / threadsPerBlock.y);

            fragmentShader<<<numBlocks, threadsPerBlock>>>(windowRows,
                                                           windowCols,
                                                           g_hostIndexDevPtr.get(),
                                                           numTriangles,
                                                           g_hostVertex2DDevPtr.get(),
                                                           g_hostVertex3DDevPtr.get(),
                                                           g_hostTriangleCross2DDevPtr.get(),
                                                           g_hostTriangleNormalDevPtr.get(),
                                                           g_hostBoundingBoxDevPtr.get(),
                                                           ls,
                                                           g_hostCharDevPtr.get());
            CUDA_CHECK(cudaGetLastError());
        }

    } // anonymous

    __host__ void loadScene(const Scene& scene)
    {
        resizeSceneDependentPtrs(scene.verticesPtr().size(), scene.indicesPtr().size());

        scene.verticesPtr().loadToDev(g_hostVertex3DDevPtr.get());
        scene.indicesPtr().loadToDev(g_hostIndexDevPtr.get());
    }

    __host__ void draw(const lightSource_t& ls)
    {
        Vector2<unsigned int> windowDim = g_hostInactiveBuf.getWindowDimensions();

        windowDim.x() = std::min(windowDim.x(), windowDim.y());
        windowDim.y() = std::min(windowDim.x(), windowDim.y());

        PerspectiveProjMatrix PPM = CCV2ScreenPPM(windowDim.y(), windowDim.x());

        resizeScreenDependentPtrs(windowDim.y(), windowDim.x());

        callVertexShader(PPM);

        CUDA_CHECK(cudaDeviceSynchronize());

        callGeometryShader();

        CUDA_CHECK(cudaDeviceSynchronize());

        callFragmentShader(ls, windowDim.y(), windowDim.x());

        g_hostCharDevPtr.loadToHost(g_hostCharHostPtr.get());
        g_hostInactiveBuf.draw(g_hostCharHostPtr.get(), windowDim.y(), windowDim.x());
        g_hostInactiveBuf.activate();
        std::swap(g_hostActiveBuf, g_hostInactiveBuf);
    }

} // Custosh::Renderer