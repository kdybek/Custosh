#include "renderer.h"

#include "internal/debug_macros.h"
#include "internal/gpu_memory.h"
#include "internal/windows_console_screen_buffer.h"

namespace Custosh::Renderer
{
    namespace
    {
        /* Auxiliary structs */
        struct triangle3D_t
        {
            Vertex3D p0;
            Vertex3D p1;
            Vertex3D p2;

            __host__ __device__ explicit triangle3D_t(
                    const Vertex3D& p0 = Vertex3D(),
                    const Vertex3D& p1 = Vertex3D(),
                    const Vertex3D& p2 = Vertex3D()
            ) : p0(p0), p1(p1), p2(p2)
            {
            }
        };

        struct triangle2D_t
        {
            Vertex2D p0;
            Vertex2D p1;
            Vertex2D p2;

            __host__ __device__ explicit triangle2D_t(
                    const Vertex2D& p0 = Vertex2D(),
                    const Vertex2D& p1 = Vertex2D(),
                    const Vertex2D& p2 = Vertex2D()
            ) : p0(p0), p1(p1), p2(p2)
            {
            }
        };

        struct boundingBox_t
        {
            float xMax;
            float xMin;
            float yMax;
            float yMin;

            __host__ __device__ explicit boundingBox_t(
                    float xMax = 0.f,
                    float xMin = 0.f,
                    float yMax = 0.f,
                    float yMin = 0.f
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
            Vertex3D coords;
            Vector3<float> normal;

            __host__ __device__ explicit fragment_t(
                    bool occupied = false,
                    const Vertex3D& coords = Vertex3D(),
                    const Vector3<float>& normal = Vector3<float>()
            ) : occupied(occupied), coords(coords), normal(normal)
            {
            }
        };

        struct triangleVariables_t
        {
            float area2x;
            boundingBox_t boundingBox;
            Vector3<float> normal;

            __host__ __device__ explicit triangleVariables_t(
                    float area2x = 0.f,
                    boundingBox_t boundingBox = boundingBox_t(),
                    Vector3<float> normal = Vector3<float>()
            ) : area2x(area2x), boundingBox(boundingBox), normal(normal)
            {
            }
        };

        struct fullTriangleInfo_t
        {
            triangle3D_t coords3D;
            triangle2D_t coords2D;
            triangleVariables_t triangleVars;

            __host__ __device__ explicit fullTriangleInfo_t(
                    triangle3D_t coords3D = triangle3D_t(),
                    triangle2D_t coords2D = triangle2D_t(),
                    triangleVariables_t triangleVars = triangleVariables_t()
            ) : coords3D(coords3D), coords2D(coords2D), triangleVars(triangleVars)
            {
            }
        };

        /* Host constants */
        constexpr unsigned int H_BASE_DEV_WSPACE_SIZE = 8;
        constexpr Vertex3D H_CCV_MIN_CORNER = {-1.f, -1.f, -1.f};
        constexpr Vertex3D H_CCV_MAX_CORNER = {1.f, 1.f, 1.f};
        constexpr float H_PM_NEAR_PLANE = 1.f;
        constexpr float H_PM_FAR_PLANE = 1000.f;
        constexpr unsigned int H_THREADS_PER_BLOCK_X = 8;
        constexpr unsigned int H_THREADS_PER_BLOCK_Y = 8;
        constexpr unsigned int H_THREADS_PER_BLOCK = H_THREADS_PER_BLOCK_X * H_THREADS_PER_BLOCK_Y;
        constexpr TransformMatrix H_IDENTITY_MATRIX = {{1.f, 0.f, 0.f, 0.f},
                                                       {0.f, 1.f, 0.f, 0.f},
                                                       {0.f, 0.f, 1.f, 0.f},
                                                       {0.f, 0.f, 0.f, 1.f}};

        /* Device constants */
        __constant__ constexpr char D_ASCII_BY_BRIGHTNESS[93] =
                R"( `.-':_,^=;><+!rc*/z?sLTv)J7(|Fi{C}fI31tlu[neoZ5Yxjya]2ESwqkP6h9d4VpOGbUAKXHm8RD#$Bg0MNWQ%&@)";
        __constant__ constexpr unsigned int D_NUM_ASCII = 92;
        __constant__ constexpr unsigned int D_THREADS_PER_BLOCK = H_THREADS_PER_BLOCK;

        /* Device global variables */
        __constant__ constinit lightSource_t g_devLightSource;

        /* @formatter:off */

        /* Host global variables */
        WindowsConsoleScreenBuffer& getFrontBuffer() { static WindowsConsoleScreenBuffer s_frontBuffer; return s_frontBuffer; }
        WindowsConsoleScreenBuffer& getBackBuffer() { static WindowsConsoleScreenBuffer s_backBuffer; return s_backBuffer; }
        HostPtr<TransformMatrix>& getTransformHostPtr() { static HostPtr<TransformMatrix> s_transformHostPtr(H_BASE_DEV_WSPACE_SIZE); return s_transformHostPtr; }
        HostPtr<char>& getFrameBufferHostPtr() { static HostPtr<char> s_frameBufferHostPtr(H_BASE_DEV_WSPACE_SIZE); return s_frameBufferHostPtr; }

        /* Device working space pointers */
        DevPtr<meshVertex_t>& getMeshVertexDevPtr() { static DevPtr<meshVertex_t> s_meshVectorDevPtr(H_BASE_DEV_WSPACE_SIZE); return s_meshVectorDevPtr; }
        DevPtr<triangleIndices_t>& getTriangleIndDevPtr() { static DevPtr<triangleIndices_t> s_triangleIndDevPtr(H_BASE_DEV_WSPACE_SIZE); return s_triangleIndDevPtr; }
        DevPtr<TransformMatrix>& getTransformDevPtr() { static DevPtr<TransformMatrix> s_transformDevPtr(H_BASE_DEV_WSPACE_SIZE); return s_transformDevPtr; }
        DevPtr<Vertex2D>& getVertex2DDevPtr() { static DevPtr<Vertex2D> s_vertex2DDevPtr(H_BASE_DEV_WSPACE_SIZE); return s_vertex2DDevPtr; }
        DevPtr<triangleVariables_t>& getTriangleVarsPtr() { static DevPtr<triangleVariables_t> s_triangleVarsPtr(H_BASE_DEV_WSPACE_SIZE); return s_triangleVarsPtr; }
        DevPtr<char>& getFrameBufferDevPtr() { static DevPtr<char> s_frameBufferDevPtr(H_BASE_DEV_WSPACE_SIZE); return s_frameBufferDevPtr; }

        /* @formatter:on */

        /* Device auxiliary functions */
        template<typename T>
        [[nodiscard]] __device__ T clamp(T a, T min, T max)
        {
            if (a < min) { return min; }
            else if (a > max) { return max; }
            else { return a; }
        }

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
            unsigned int idx = ceil(brightness * static_cast<float>(D_NUM_ASCII - 1));
            return D_ASCII_BY_BRIGHTNESS[idx];
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

            float w0 = cross2D(triangle2D.p1, triangle2D.p2, p);
            float w1 = cross2D(triangle2D.p2, triangle2D.p0, p);
            float w2 = cross2D(triangle2D.p0, triangle2D.p1, p);

            if (w0 == 0.f && isBottomOrRight(triangle2D.p1, triangle2D.p2)) { return false; }
            if (w1 == 0.f && isBottomOrRight(triangle2D.p2, triangle2D.p0)) { return false; }
            if (w2 == 0.f && isBottomOrRight(triangle2D.p0, triangle2D.p1)) { return false; }

            barycentricCoords.alpha = w0 / triangleArea2x;
            barycentricCoords.beta = w1 / triangleArea2x;
            barycentricCoords.gamma = w2 / triangleArea2x;

            return (w0 >= 0.f && w1 >= 0.f && w2 >= 0.f);
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

        [[nodiscard]] __device__ float fragmentBrightness(const fragment_t& p, const lightSource_t& ls)
        {
            if (!p.occupied) { return 0.f; }

            float distSq = distanceSq(p.coords, ls.coords);
            auto pointToLightSourceVec = Vector3<float>(ls.coords - p.coords);
            auto pointToLightSourceVecNorm = Vector3<float>(pointToLightSourceVec.normalized());
            float cos = pointToLightSourceVecNorm.dot(p.normal);

            return clamp(max(cos, 0.f) * ls.intensity / distSq, 0.f, 1.f);
        }

        // The vertices are in clockwise order, but we're looking from 0 towards positive z values.
        [[nodiscard]] __device__ Vector3<float> triangleNormal(const triangle3D_t& triangle3D)
        {
            Vector3<float> normal = Vector3<float>(triangle3D.p2 - triangle3D.p0).cross(
                    Vector3<float>(triangle3D.p1 - triangle3D.p0));

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
                                                            const meshVertex_t* meshVertexPtr)
        {
            return triangle3D_t(meshVertexPtr[triangleIndices.p0].coords,
                                meshVertexPtr[triangleIndices.p1].coords,
                                meshVertexPtr[triangleIndices.p2].coords);
        }

        /* Kernels */
        __global__ void vertexShader(meshVertex_t* meshVertexPtr,
                                     unsigned int numVertices,
                                     const TransformMatrix* transformMatPtr,
                                     PerspectiveProjectionMatrix ppm,
                                     Vertex2D* vertex2DPtr)
        {
            const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

            if (i >= numVertices) { return; }

            meshVertex_t meshVertex = meshVertexPtr[i];

            Vector4<float> updatedVertex4D = Vector4<float>(transformMatPtr[meshVertex.meshIdx] *
                                                            meshVertex.coords.toHomogeneous()).normalizeW();

            Vector4<float> vertex4DPerspective = Vector4<float>(ppm * updatedVertex4D).normalizeW();

            meshVertexPtr[i] = meshVertex_t({updatedVertex4D.x(), updatedVertex4D.y(), updatedVertex4D.z()},
                                            meshVertex.meshIdx);

            vertex2DPtr[i] = {vertex4DPerspective.x(), vertex4DPerspective.y()};
        }

        __global__ void geometryShader(const triangleIndices_t* triangleIndPtr,
                                       unsigned int numTriangles,
                                       const Vertex2D* vertex2DPtr,
                                       const meshVertex_t* meshVertexPtr,
                                       triangleVariables_t* triangleVarsPtr)
        {
            const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

            if (i >= numTriangles) { return; }

            triangleIndices_t tIndRef = triangleIndPtr[i];
            auto& tVarsRef = triangleVarsPtr[i];

            triangle2D_t triangle2D = getTriangle2D(tIndRef, vertex2DPtr);
            triangle3D_t triangle3D = getTriangle3D(tIndRef, meshVertexPtr);

            tVarsRef.area2x = cross2D(triangle2D.p0, triangle2D.p1, triangle2D.p2);
            tVarsRef.normal = triangleNormal(triangle3D);
            tVarsRef.boundingBox = findBounds(triangle2D);
        }

        __global__ void fragmentShader(unsigned int rows,
                                       unsigned int cols,
                                       const triangleIndices_t* indexPtr,
                                       unsigned int numTriangles,
                                       const Vertex2D* vertex2DPtr,
                                       const meshVertex_t* meshVertexPtr,
                                       const triangleVariables_t* tVarsPtr,
                                       char* frameBuffer)
        {
            __shared__ char sh_memory[D_THREADS_PER_BLOCK * sizeof(fullTriangleInfo_t)];

            auto* sh_triangleInfoPtr = reinterpret_cast<fullTriangleInfo_t*>(sh_memory);

            const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
            const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
            const unsigned int sharedMemIdx = threadIdx.x * blockDim.y + threadIdx.y;
            const unsigned int blocksToLoad = (numTriangles + D_THREADS_PER_BLOCK - 1) / D_THREADS_PER_BLOCK;

            fragment_t fragment;

            for (unsigned int i = 0; i < blocksToLoad; ++i) {
                const unsigned int globalMemIdx = i * D_THREADS_PER_BLOCK + sharedMemIdx;

                if (globalMemIdx < numTriangles) {
                    triangleIndices_t triangleIndices = indexPtr[globalMemIdx];
                    auto& sh_tInfoRef = sh_triangleInfoPtr[sharedMemIdx];

                    sh_tInfoRef.coords3D = getTriangle3D(triangleIndices, meshVertexPtr);
                    sh_tInfoRef.coords2D = getTriangle2D(triangleIndices, vertex2DPtr);
                    sh_tInfoRef.triangleVars = tVarsPtr[globalMemIdx];
                }

                __syncthreads();

                if (y < rows && x < cols) {
                    const unsigned int trianglesInBlock = min(numTriangles - i * D_THREADS_PER_BLOCK,
                                                              D_THREADS_PER_BLOCK);

                    for (unsigned int j = 0; j < trianglesInBlock; ++j) {
                        fullTriangleInfo_t tInfo = sh_triangleInfoPtr[j];
                        barycentricCoords_t bc;

                        if (tInfo.triangleVars.area2x == 0.f) { continue; }

                        if (inTriangle(tInfo.coords2D,
                                       tInfo.triangleVars.boundingBox,
                                       Vertex2D({static_cast<float>(x), static_cast<float>(y)}),
                                       tInfo.triangleVars.area2x,
                                       bc)) {
                            Vertex3D projectedPoint = getCartesianCoords(tInfo.coords3D, bc);

                            if (!fragment.occupied || fragment.coords.z() > projectedPoint.z()) {
                                fragment.occupied = true;
                                fragment.coords = projectedPoint;
                                fragment.normal = tInfo.triangleVars.normal;
                            }
                        }
                    }
                }

                __syncthreads();
            }

            if (y < rows && x < cols) {
                frameBuffer[y * cols + x] = brightnessToASCII(fragmentBrightness(fragment, g_devLightSource));
            }
        }

        /* Host auxiliary functions */
        [[nodiscard]] __host__ PerspectiveProjectionMatrix CCV2ScreenPPM(unsigned int screenRows,
                                                                         unsigned int screenCols)
        {
            return {PerspectiveMatrix(H_PM_NEAR_PLANE, H_PM_FAR_PLANE),
                    OrthographicProjectionMatrix(H_CCV_MIN_CORNER,
                                                 H_CCV_MAX_CORNER,
                                                 {0.f, 0.f, 0.f},
                                                 {static_cast<float>(screenCols), static_cast<float>(screenRows),
                                                  0.f})};
        }

        __host__ void resizeSceneDependentPtrs(unsigned int numVertices,
                                               unsigned int numTriangles,
                                               unsigned int numMeshes)
        {
            getMeshVertexDevPtr().resizeAndDiscardData(numVertices);
            getVertex2DDevPtr().resizeAndDiscardData(numVertices);

            getTriangleIndDevPtr().resizeAndDiscardData(numTriangles);
            getTriangleVarsPtr().resizeAndDiscardData(numTriangles);

            getTransformDevPtr().resizeAndDiscardData(numMeshes);
            getTransformHostPtr().resizeAndDiscardData(numMeshes);
        }

        __host__ void resizeScreenDependentPtrs(unsigned int screenRows,
                                                unsigned int screenCols)
        {
            getFrameBufferHostPtr().resizeAndDiscardData(screenRows * screenCols);
            getFrameBufferDevPtr().resizeAndDiscardData(screenRows * screenCols);
        }

        __host__ void callVertexShader(const PerspectiveProjectionMatrix& ppm)
        {
            unsigned int numVertices = getMeshVertexDevPtr().size();

            unsigned int threadsPerBlock = std::min(numVertices, static_cast<unsigned int>(H_THREADS_PER_BLOCK));
            unsigned int numBlocks = (numVertices + threadsPerBlock - 1) / threadsPerBlock;

            vertexShader<<<numBlocks, threadsPerBlock>>>(getMeshVertexDevPtr().get(),
                                                         numVertices,
                                                         getTransformDevPtr().get(),
                                                         ppm,
                                                         getVertex2DDevPtr().get());
            CUSTOSH_CUDA_CHECK(cudaGetLastError());
        }

        __host__ void callGeometryShader()
        {
            unsigned int numTriangles = getTriangleIndDevPtr().size();

            unsigned int threadsPerBlock = std::min(numTriangles, H_THREADS_PER_BLOCK);
            unsigned int numBlocks = (numTriangles + threadsPerBlock - 1) / threadsPerBlock;

            geometryShader<<<numBlocks, threadsPerBlock>>>(getTriangleIndDevPtr().get(),
                                                           numTriangles,
                                                           getVertex2DDevPtr().get(),
                                                           getMeshVertexDevPtr().get(),
                                                           getTriangleVarsPtr().get());
            CUSTOSH_CUDA_CHECK(cudaGetLastError());
        }

        __host__ void callFragmentShader(unsigned int screenRows,
                                         unsigned int screenCols)
        {
            unsigned int numTriangles = getTriangleIndDevPtr().size();

            dim3 threadsPerBlock(H_THREADS_PER_BLOCK_X, H_THREADS_PER_BLOCK_Y);
            dim3 numBlocks((screenCols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                           (screenRows + threadsPerBlock.y - 1) / threadsPerBlock.y);

            fragmentShader<<<numBlocks, threadsPerBlock>>>(screenRows,
                                                           screenCols,
                                                           getTriangleIndDevPtr().get(),
                                                           numTriangles,
                                                           getVertex2DDevPtr().get(),
                                                           getMeshVertexDevPtr().get(),
                                                           getTriangleVarsPtr().get(),
                                                           getFrameBufferDevPtr().get());
            CUSTOSH_CUDA_CHECK(cudaGetLastError());
        }

        __host__ void resetTransformMatrices()
        {
            for (unsigned int i = 0; i < getTransformHostPtr().size(); ++i) {
                getTransformHostPtr().get()[i] = H_IDENTITY_MATRIX;
            }

            getTransformHostPtr().loadToDev(getTransformDevPtr().get(), getTransformHostPtr().size());
        }

        __host__ void setLightSource(const lightSource_t& ls)
        {
            CUSTOSH_CUDA_CHECK(cudaMemcpyToSymbol(g_devLightSource, &ls, sizeof(lightSource_t)));
        }

        __host__ void renderingPipeline(unsigned int screenRows,
                                        unsigned int screenCols,
                                        const PerspectiveProjectionMatrix& ppm)
        {
            callVertexShader(ppm);

            CUSTOSH_CUDA_CHECK(cudaDeviceSynchronize());

            callGeometryShader();

            CUSTOSH_CUDA_CHECK(cudaDeviceSynchronize());

            callFragmentShader(screenRows, screenCols);

            // Host and device will synchronize when copying the frame buffer to host,
            // but I synchronize them here to accurately measure time.
            CUSTOSH_DEBUG_CALL(CUSTOSH_CUDA_CHECK(cudaDeviceSynchronize()));
        }

        __host__ void drawFrameBuffer(unsigned int screenRows,
                                      unsigned int screenCols)
        {
            getBackBuffer().draw(getFrameBufferHostPtr().get(), screenRows, screenCols);
            getBackBuffer().activate();
            std::swap(getFrontBuffer(), getBackBuffer());
        }

    } // anonymous

    __host__ void loadScene(const Scene& scene)
    {
        resizeSceneDependentPtrs(scene.numVertices(), scene.numTriangles(), scene.numMeshes());

        scene.loadVerticesToDev(getMeshVertexDevPtr().get());
        scene.loadTrianglesToDev(getTriangleIndDevPtr().get());

        resetTransformMatrices();

        setLightSource(scene.lightSource());
    }

    __host__ void loadTransformMatrix(const TransformMatrix& tm, unsigned int meshIdx)
    {
        if (meshIdx >= getTransformHostPtr().size()) { throw CustoshException("invalid mesh index"); }

        getTransformHostPtr().get()[meshIdx] = tm;

        getTransformHostPtr().loadToDev(getTransformDevPtr().get(), getTransformHostPtr().size());
    }

    __host__ void transformVerticesAndDraw()
    {
        Vector2<unsigned int> screenDim = getBackBuffer().getWindowDimensions();

        screenDim.x() = std::min(screenDim.x(), screenDim.y());
        screenDim.y() = std::min(screenDim.x(), screenDim.y());

        if (screenDim.x() == 0 || screenDim.y() == 0) { return; }

        PerspectiveProjectionMatrix ppm = CCV2ScreenPPM(screenDim.y(), screenDim.x());

        resizeScreenDependentPtrs(screenDim.y(), screenDim.x());

        CUSTOSH_DEBUG_LOG_TIME(renderingPipeline(screenDim.y(), screenDim.x(), ppm), "rendering pipeline");

        // This is slow, but if I want to draw in the terminal there is no way to bypass the CPU as far as I know.
        CUSTOSH_DEBUG_LOG_TIME(
                getFrameBufferDevPtr().loadToHost(getFrameBufferHostPtr().get(), screenDim.y() * screenDim.x()),
                "fetch frame buffer from GPU");

        // This is currently the biggest bottleneck, but works fine most of the time.
        CUSTOSH_DEBUG_LOG_TIME(drawFrameBuffer(screenDim.y(), screenDim.x()), "write to terminal");
    }

} // Custosh::Renderer