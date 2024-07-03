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

        /* Device working space pointers */
        DevPtr<Vector2<float>> g_hostVertex2DDevPtr(BASE_DEV_WSPACE_SIZE);
        DevPtr<boundingBox_t> g_hostBoundingBoxDevPtr(BASE_DEV_WSPACE_SIZE);
        DevPtr<float> g_hostTriangleCross2DDevPtr(BASE_DEV_WSPACE_SIZE);
        DevPtr<Vector3<float>> g_hostTriangleNormalDevPtr(BASE_DEV_WSPACE_SIZE);
        HostDevPtr<char> g_hostCharPtr(BASE_DEV_WSPACE_SIZE);

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
            if (p.x() >= boundingBox.xMin && p.x() <= boundingBox.xMax &&
                p.y() >= boundingBox.yMin && p.y() <= boundingBox.yMax) {
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
        getCartesianCoords(const triangle3D_t& triangle3D, const barycentricCoords_t& bc)
        {
            return {triangle3D.p0.x() * bc.alpha + triangle3D.p1.x() * bc.beta + triangle3D.p2.x() * bc.gamma,
                    triangle3D.p0.y() * bc.alpha + triangle3D.p1.y() * bc.beta + triangle3D.p2.y() * bc.gamma,
                    triangle3D.p0.z() * bc.alpha + triangle3D.p1.z() * bc.beta + triangle3D.p2.z() * bc.gamma};
        }

        [[nodiscard]] __device__ float distanceSq(const Vector3<float>& a, const Vector3<float>& b)
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

        [[nodiscard]] __device__ float pointBrightness(const fragment_t& p, const lightSource_t& ls)
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

        /* Kernels */
        // TODO: vertex manipulation (translation, rotation, etc.)
        __global__ void vertexShader(const Vector3<float>* vertex3DPtr,
                                     unsigned int numVertices,
                                     PerspectiveProjMatrix ppm,
                                     Vector2<float>* vertex2DPtr)
        {
            const unsigned int i = threadIdx.x;

            if (i >= numVertices) { return; }

            vertex2DPtr[i] = applyPerspective(vertex3DPtr[i], ppm);
        }

        __global__ void populateTriangleParams(triangleIndices_t* indexPtr,
                                               unsigned int numTriangles,
                                               const Vector2<float>* vertex2DPtr,
                                               const Vector3<float>* vertex3DPtr,
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
                                       const Vector2<float>* vertex2DPtr,
                                       const Vector3<float>* vertex3DPtr,
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
                               Vector2<float>({static_cast<float>(x), static_cast<float>(y)}),
                               triangleArea2x,
                               bc)) {
                    Vector3<float> projectedPoint = getCartesianCoords(triangle3D, bc);

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
        [[nodiscard]] __host__ Vector2<unsigned int> getWindowDimensions()
        {
            HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
            if (hConsole == INVALID_HANDLE_VALUE) {
                throw CustoshException("error getting console handle");
            }

            CONSOLE_SCREEN_BUFFER_INFO csbi;
            if (!GetConsoleScreenBufferInfo(hConsole, &csbi)) {
                throw CustoshException("error getting console screen buffer info");
            }

            unsigned int rows = csbi.srWindow.Bottom - csbi.srWindow.Top + 1;
            unsigned int cols = csbi.srWindow.Right - csbi.srWindow.Left + 1;

            return {cols, rows};
        }

        [[nodiscard]] __host__ PerspectiveProjMatrix CCV2ScreenPPM(unsigned int screenRows, unsigned int screenCols)
        {
            return {PerspectiveMatrix(PM_NEAR_PLANE, PM_FAR_PLANE),
                    OrtProjMatrix(CCV_MIN_CORNER,
                                  CCV_MAX_CORNER,
                                  {0.f, 0.f, 0.f},
                                  {static_cast<float>(screenCols), static_cast<float>(screenRows), 0.f})};
        }

        __host__ void resizePtrs(unsigned int numVertices,
                                 unsigned int numTriangles,
                                 unsigned int windowRows,
                                 unsigned int windowCols)
        {
            if (numVertices > g_hostVertex2DDevPtr.size()) {
                g_hostVertex2DDevPtr.resizeAndDiscardData(numVertices);
            }

            if (numTriangles > g_hostBoundingBoxDevPtr.size()) {
                g_hostBoundingBoxDevPtr.resizeAndDiscardData(numTriangles);
                g_hostTriangleCross2DDevPtr.resizeAndDiscardData(numTriangles);
                g_hostTriangleNormalDevPtr.resizeAndDiscardData(numTriangles);
            }

            if (windowRows * windowCols > g_hostCharPtr.size()) {
                g_hostCharPtr.resizeAndDiscardData(windowRows * windowCols);
            }
        }

        __host__ void callVertexShaderKernel(const Mesh& mesh, const PerspectiveProjMatrix& PPM)
        {
            const Vector3<float>* vertex3DPtr = mesh.hostDevVerticesPtr().devPtr();
            unsigned int numVertices = mesh.hostDevVerticesPtr().size();

            unsigned int threadsPerBlock = std::min(numVertices, static_cast<unsigned int>(MAX_THREADS_PER_BLOCK));
            unsigned int numBlocks = (numVertices + threadsPerBlock - 1) / threadsPerBlock;

            vertexShader<<<numBlocks, threadsPerBlock>>>(vertex3DPtr,
                                                         numVertices,
                                                         PPM,
                                                         g_hostVertex2DDevPtr.get());
            CUDA_CHECK(cudaGetLastError());
        }

        __host__ void callPopulateTriangleParamsKernel(const Mesh& mesh)
        {
            const Vector3<float>* vertex3DPtr = mesh.hostDevVerticesPtr().devPtr();
            triangleIndices_t* indexPtr = mesh.hostDevIndicesPtr().devPtr();
            unsigned int numTriangles = mesh.hostDevIndicesPtr().size();

            unsigned int threadsPerBlock = std::min(numTriangles, static_cast<unsigned int>(MAX_THREADS_PER_BLOCK));
            unsigned int numBlocks = (numTriangles + threadsPerBlock - 1) / threadsPerBlock;

            populateTriangleParams<<<numBlocks, threadsPerBlock>>>(indexPtr,
                                                                   numTriangles,
                                                                   g_hostVertex2DDevPtr.get(),
                                                                   vertex3DPtr,
                                                                   g_hostTriangleCross2DDevPtr.get(),
                                                                   g_hostTriangleNormalDevPtr.get(),
                                                                   g_hostBoundingBoxDevPtr.get());
            CUDA_CHECK(cudaGetLastError());
        }

        __host__ void callFragmentShaderKernel(const Mesh& mesh,
                                               const lightSource_t& ls,
                                               unsigned int windowRows,
                                               unsigned int windowCols)
        {
            const Vector3<float>* vertex3DPtr = mesh.hostDevVerticesPtr().devPtr();
            triangleIndices_t* indexPtr = mesh.hostDevIndicesPtr().devPtr();
            unsigned int numTriangles = mesh.hostDevIndicesPtr().size();

            dim3 threadsPerBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
            dim3 numBlocks((windowCols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                           (windowRows + threadsPerBlock.y - 1) / threadsPerBlock.y);

            fragmentShader<<<numBlocks, threadsPerBlock>>>(windowRows,
                                                           windowCols,
                                                           indexPtr,
                                                           numTriangles,
                                                           g_hostVertex2DDevPtr.get(),
                                                           vertex3DPtr,
                                                           g_hostTriangleCross2DDevPtr.get(),
                                                           g_hostTriangleNormalDevPtr.get(),
                                                           g_hostBoundingBoxDevPtr.get(),
                                                           ls,
                                                           g_hostCharPtr.devPtr());
            CUDA_CHECK(cudaGetLastError());
        }

    } // anonymous

    __host__ void drawMesh(const Mesh& mesh, const lightSource_t& ls)
    {
        Vector2<unsigned int> windowDim = getWindowDimensions();

        windowDim.x() = std::min(windowDim.x(), windowDim.y());
        windowDim.y() = std::min(windowDim.x(), windowDim.y());

        PerspectiveProjMatrix PPM = CCV2ScreenPPM(windowDim.y(), windowDim.x());

        resizePtrs(mesh.hostDevVerticesPtr().size(), mesh.hostDevIndicesPtr().size(), windowDim.y(), windowDim.x());

        mesh.hostDevVerticesPtr().loadToDev();
        mesh.hostDevIndicesPtr().loadToDev();

        callVertexShaderKernel(mesh, PPM);

        CUDA_CHECK(cudaDeviceSynchronize());

        callPopulateTriangleParamsKernel(mesh);

        CUDA_CHECK(cudaDeviceSynchronize());

        callFragmentShaderKernel(mesh, ls, windowDim.y(), windowDim.x());

        g_hostCharPtr.loadToHost();
        g_hostInactiveBuf.draw(g_hostCharPtr.hostPtr(), windowDim.y(), windowDim.x());
        g_hostInactiveBuf.activate();
        std::swap(g_hostActiveBuf, g_hostInactiveBuf);
    }

} // Custosh::Renderer