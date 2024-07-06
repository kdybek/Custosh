#include <iostream>
#include <chrono>
#include <thread>
#include <algorithm>

#include "Utility.cuh"
#include "Renderer.cuh"

using namespace Custosh;

int main()
{
    std::vector<Vector3<float>> cube1Ver;
    cube1Ver.push_back({-1.5, -0.5, 2});
    cube1Ver.push_back({-0.5, -0.5, 2});
    cube1Ver.push_back({-1.5, 0.5, 2});
    cube1Ver.push_back({-0.5, 0.5, 2});

    cube1Ver.push_back({-1.5, -0.5, 3});
    cube1Ver.push_back({-0.5, -0.5, 3});
    cube1Ver.push_back({-1.5, 0.5, 3});
    cube1Ver.push_back({-0.5, 0.5, 3});

    std::vector<Vector3<float>> cube2Ver;
    cube2Ver.push_back({0.5, -0.5, 2});
    cube2Ver.push_back({1.5, -0.5, 2});
    cube2Ver.push_back({0.5, 0.5, 2});
    cube2Ver.push_back({1.5, 0.5, 2});

    cube2Ver.push_back({0.5, -0.5, 3});
    cube2Ver.push_back({1.5, -0.5, 3});
    cube2Ver.push_back({0.5, 0.5, 3});
    cube2Ver.push_back({1.5, 0.5, 3});

    std::vector<triangleIndices_t> cubeInd;
    cubeInd.emplace_back(0, 2, 3);
    cubeInd.emplace_back(3, 1, 0);
    cubeInd.emplace_back(4, 5, 7);
    cubeInd.emplace_back(7, 6, 4);

    cubeInd.emplace_back(0, 2, 6);
    cubeInd.emplace_back(6, 4, 0);
    cubeInd.emplace_back(1, 3, 7);
    cubeInd.emplace_back(7, 5, 1);

    cubeInd.emplace_back(2, 6, 7);
    cubeInd.emplace_back(7, 3, 2);
    cubeInd.emplace_back(0, 1, 5);
    cubeInd.emplace_back(5, 4, 0);

    Mesh cube1(cube1Ver, cubeInd);
    Mesh cube2(cube2Ver, cubeInd);

    lightSource_t ls({0, 0, 0}, 0.7);

    float rotationAngle1 = degreesToRadians(1.5);
    float rotationAngle2 = degreesToRadians(1);
    float rotationAngle3 = degreesToRadians(0.5);

    Vector3<float> rotationVec1 = {0, 1, 0};
    Vector3<float> rotationVec2 = {1, 0, 0};
    Vector3<float> rotationVec3 = {0, 0, 1};

    Vertex3D origin1 = {-1, 0, 2.5};
    Vertex3D origin2 = {1, 0, 2.5};

    Scene scene;

    scene.add(cube1);
    scene.add(cube2);

    Renderer::loadScene(scene);

    TransformMatrix rotationMat1 = DecentralizedTransformMatrix(RotationMatrix(rotationVec1, rotationAngle1), origin1);
    TransformMatrix rotationMat2 = DecentralizedTransformMatrix(RotationMatrix(rotationVec2, rotationAngle2), origin1);
    TransformMatrix rotationMat3 = DecentralizedTransformMatrix(RotationMatrix(rotationVec3, rotationAngle3), origin1);

    TransformMatrix rotationMat4 = DecentralizedTransformMatrix(RotationMatrix(rotationVec2, rotationAngle1), origin2);
    TransformMatrix rotationMat5 = DecentralizedTransformMatrix(RotationMatrix(rotationVec3, rotationAngle2), origin2);
    TransformMatrix rotationMat6 = DecentralizedTransformMatrix(RotationMatrix(rotationVec1, rotationAngle3), origin2);

    Renderer::loadTransformMatrix(rotationMat1 * rotationMat2 * rotationMat3, 0);

    Renderer::loadTransformMatrix(rotationMat4 * rotationMat5 * rotationMat6, 1);

    while (true) {
        auto start = std::chrono::high_resolution_clock::now();

        Renderer::draw(ls);

        auto end = std::chrono::high_resolution_clock::now();

        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::this_thread::sleep_for(std::chrono::milliseconds(std::max((long long) 0, 15 - elapsed.count())));
    }
}

// TODO: logging