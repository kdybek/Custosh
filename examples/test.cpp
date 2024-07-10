/* Performance testing */

#include <chrono>
#include <thread>
#include <random>

#include <custosh.h>

using namespace Custosh;

// Checking for static initialization order fiasco while we're at it.
int realMain()
{
    LoggerManager::addLogger(std::make_unique<ConsoleLogger>(ConsoleLogger()));

    std::vector<Vertex3D> cube1Ver;
    cube1Ver.push_back({-0.5f, -0.5f, 3.f});
    cube1Ver.push_back({0.5f, -0.5f, 3.f});
    cube1Ver.push_back({-0.5f, 0.5f, 3.f});
    cube1Ver.push_back({0.5f, 0.5f, 3.f});

    cube1Ver.push_back({-0.5f, -0.5f, 4.f});
    cube1Ver.push_back({0.5f, -0.5f, 4.f});
    cube1Ver.push_back({-0.5f, 0.5f, 4.f});
    cube1Ver.push_back({0.5f, 0.5f, 4.f});

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

    float rotationAngle1 = degreesToRadians(1.5f);
    float rotationAngle2 = degreesToRadians(1.f);
    float rotationAngle3 = degreesToRadians(0.5f);

    Vector3<float> rotationVec1 = {0.f, 1.f, 0.f};
    Vector3<float> rotationVec2 = {1.f, 0.f, 0.f};
    Vector3<float> rotationVec3 = {0.f, 0.f, 1.f};

    Vertex3D origin1 = {0.f, 0.f, 3.5f};

    lightSource_t ls({1.f, 0.f, 0.f}, 0.9f);

    Scene scene(ls);

    unsigned int numCubes = 100000;

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.f, 1.f);

    for (unsigned int i = 0; i < numCubes; ++i) {
        std::vector<Vertex3D> res;
        float xRandomNum = dis(gen);
        float yRandomNum = dis(gen);
        float zRandomNum = dis(gen);

        for (const auto& vertex: cube1Ver) {
            res.push_back({vertex.x() + xRandomNum, vertex.y() + yRandomNum, vertex.z() + zRandomNum});
        }

        scene.add(Mesh(res, cubeInd));
    }

    Renderer::loadScene(scene);

    TransformMatrix rotationMat1 = DecentralizedTransformMatrix(RotationMatrix(rotationVec1, rotationAngle1),
                                                                origin1);
    TransformMatrix rotationMat2 = DecentralizedTransformMatrix(RotationMatrix(rotationVec2, rotationAngle2),
                                                                origin1);
    TransformMatrix rotationMat3 = DecentralizedTransformMatrix(RotationMatrix(rotationVec3, rotationAngle3),
                                                                origin1);

    for (unsigned int i = 0; i < numCubes; ++i) {
        Renderer::loadTransformMatrix(rotationMat1 * rotationMat2 * rotationMat3, i);
    }

    Renderer::transformVerticesAndDraw();

    return 42;
}

namespace
{
    int a = realMain();
}

int main()
{
}