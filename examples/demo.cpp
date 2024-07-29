/* API showcase */

#include <custosh.h>

#include <chrono>
#include <thread>

using namespace Custosh;

void rotatingCubesExample()
{
    /* *
     * If you define CMAKE_BUILD_TYPE as "Debug" the library will log performance information.
     * Warning: time measuring has a noticeable impact on performance.
     * You can specify where to write the logs by adding Loggers.
     * The FileLogger truncates specified files.
     * */
    LoggerManager::addLogger(std::make_unique<ConsoleLogger>());
    LoggerManager::addLogger(std::make_unique<FileLogger>("log.txt"));

    /* *
     * To define a mesh you need to specify the locations of vertices and the triangle vertex indices.
     * We are looking towards increasing z values with the view frustum.
     * Only triangles whose vertices are in a clockwise order when projected onto the near plane of the view frustum will be rendered.
     * For more parameter info consult the constants section in renderer.cu.
     * */
    std::vector<Vertex3D> cube1Ver;
    cube1Ver.push_back({-1.5f, -0.5f, 2.f});
    cube1Ver.push_back({-0.5f, -0.5f, 2.f});
    cube1Ver.push_back({-1.5f, 0.5f, 2.f});
    cube1Ver.push_back({-0.5f, 0.5f, 2.f});

    cube1Ver.push_back({-1.5f, -0.5f, 3.f});
    cube1Ver.push_back({-0.5f, -0.5f, 3.f});
    cube1Ver.push_back({-1.5f, 0.5f, 3.f});
    cube1Ver.push_back({-0.5f, 0.5f, 3.f});

    std::vector<Vertex3D> cube2Ver;
    cube2Ver.push_back({0.5f, -0.5f, 2.f});
    cube2Ver.push_back({1.5f, -0.5f, 2.f});
    cube2Ver.push_back({0.5f, 0.5f, 2.f});
    cube2Ver.push_back({1.5f, 0.5f, 2.f});

    cube2Ver.push_back({0.5f, -0.5f, 3.f});
    cube2Ver.push_back({1.5f, -0.5f, 3.f});
    cube2Ver.push_back({0.5f, 0.5f, 3.f});
    cube2Ver.push_back({1.5f, 0.5f, 3.f});

    std::vector<triangleIndices_t> cubeInd;
    cubeInd.emplace_back(0, 1, 2);
    cubeInd.emplace_back(2, 1, 3);
    cubeInd.emplace_back(4, 5, 0);
    cubeInd.emplace_back(0, 5, 1);

    cubeInd.emplace_back(6, 7, 4);
    cubeInd.emplace_back(4, 7, 5);
    cubeInd.emplace_back(2, 3, 6);
    cubeInd.emplace_back(6, 3, 7);

    cubeInd.emplace_back(1, 5, 3);
    cubeInd.emplace_back(3, 5, 7);
    cubeInd.emplace_back(4, 0, 6);
    cubeInd.emplace_back(6, 0, 2);

    Mesh cube1(cube1Ver, cubeInd);
    Mesh cube2(cube2Ver, cubeInd);

    float rotationAngle1 = degreesToRadians(1.5f);
    float rotationAngle2 = degreesToRadians(1.f);
    float rotationAngle3 = degreesToRadians(0.5f);

    Vector3<float> rotationVec1 = {0.f, 1.f, 0.f};
    Vector3<float> rotationVec2 = {1.f, 0.f, 0.f};
    Vector3<float> rotationVec3 = {0.f, 0.f, 1.f};

    Vertex3D origin1 = {-1.f, 0.f, 2.5f};
    Vertex3D origin2 = {1.f, 0.f, 2.5f};

    /* *
     * Currently there can only be one light source per scene.
     * Shadows are not being calculated yet.
     * */
    lightSource_t ls({1.f, 0.f, 0.f}, 0.7f);

    Scene scene(ls);

    unsigned int cube1Idx = scene.add(cube1);
    unsigned int cube2Idx = scene.add(cube2);

    Renderer::loadScene(scene);

    TransformMatrix rotationMat1 = DecentralizedTransformMatrix(RotationMatrix(rotationVec1, rotationAngle1),
                                                                origin1);
    TransformMatrix rotationMat2 = DecentralizedTransformMatrix(RotationMatrix(rotationVec2, rotationAngle2),
                                                                origin1);
    TransformMatrix rotationMat3 = DecentralizedTransformMatrix(RotationMatrix(rotationVec3, rotationAngle3),
                                                                origin1);

    TransformMatrix rotationMat4 = DecentralizedTransformMatrix(RotationMatrix(rotationVec2, rotationAngle1),
                                                                origin2);
    TransformMatrix rotationMat5 = DecentralizedTransformMatrix(RotationMatrix(rotationVec3, rotationAngle2),
                                                                origin2);
    TransformMatrix rotationMat6 = DecentralizedTransformMatrix(RotationMatrix(rotationVec1, rotationAngle3),
                                                                origin2);

    /* *
     * You can specify a transformation which will be applied to the mesh vertices in the vertex shader.
     * The transformation matrices are 4x4 to be consistent with the perspective projection matrix.
     * */
    Renderer::loadTransformMatrix(rotationMat1 * rotationMat2 * rotationMat3, cube1Idx);

    Renderer::loadTransformMatrix(rotationMat4 * rotationMat5 * rotationMat6, cube2Idx);

    unsigned int numFrames = 500;
    unsigned int msPerFrame = 15;

    for (unsigned int i = 0; i < numFrames; ++i) {
        auto start = std::chrono::high_resolution_clock::now();

        Renderer::transformVerticesAndDraw();

        auto end = std::chrono::high_resolution_clock::now();

        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::this_thread::sleep_for(std::chrono::milliseconds(std::max(static_cast<long long>(0), msPerFrame - elapsed.count())));
    }
}

int main()
{
    try {
        rotatingCubesExample();
    }
    catch (const CustoshException& e) {
        LoggerManager::log(LogLevel::Error, e.what());
    }
}
