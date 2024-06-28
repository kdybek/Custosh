#include <iostream>
#include <cmath>
#include <chrono>
#include <thread>
#include <algorithm>

#include "Utility.h"
#include "Renderer.h"
#include "Model.h"
#include "WindowsConsoleScreenBuffer.h"

using namespace Custosh;

int main()
{
    std::vector<Vector3<float>> cubeVer;
    cubeVer.push_back({-0.5, -0.5, 1.5});
    cubeVer.push_back({0.5, -0.5, 1.5});
    cubeVer.push_back({-0.5, 0.5, 1.5});
    cubeVer.push_back({0.5, 0.5, 1.5});

    cubeVer.push_back({-0.5, -0.5, 2.5});
    cubeVer.push_back({0.5, -0.5, 2.5});
    cubeVer.push_back({-0.5, 0.5, 2.5});
    cubeVer.push_back({0.5, 0.5, 2.5});

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

    Model cube(cubeVer, cubeInd);

    ResizableMatrix<pixel_t> screen(100, 100);
    OrtProjMatrix opm({-1, -1, 1},
                      {1, 1, 10},
                      {0, 0, 0},
                      {100, 100, 0});

    PerspectiveMatrix pm(1, 10);
    PPM ppm(pm, opm);

    lightSource_t ls = {.coords = {0, 0, 0}};

    float rotationAngle1 = degreesToRadians(3);

    float rotationAngle2 = degreesToRadians(2);

    float rotationAngle3 = degreesToRadians(1);

    Vector3<float> rotationVec1 = {0, 1, 0};

    Vector3<float> rotationVec2 = {1, 0, 0};

    Vector3<float> rotationVec3 = {0, 0, 1};

    WindowsConsoleScreenBuffer buf1;
    WindowsConsoleScreenBuffer buf2;

    while (true) {
        auto start = std::chrono::high_resolution_clock::now();

        cube.rotate({0, 0, 2}, rotationAngle1, rotationVec1);
        cube.rotate({0, 0, 2}, rotationAngle2, rotationVec2);
        cube.rotate({0, 0, 2}, rotationAngle3, rotationVec3);

        Renderer::rasterizeModel(cube, screen, ppm);
        BrightnessMap bMap = Renderer::getBrightnessMap(screen, ls);

        buf1.draw(bMap);
        buf1.activate();
        Renderer::clearScreen(screen);

        auto end = std::chrono::high_resolution_clock::now();

        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::this_thread::sleep_for(std::chrono::milliseconds(std::max((long long)0, 30 - elapsed.count())));

        start = std::chrono::high_resolution_clock::now();

        cube.rotate({0, 0, 2}, rotationAngle1, rotationVec1);
        cube.rotate({0, 0, 2}, rotationAngle2, rotationVec2);
        cube.rotate({0, 0, 2}, rotationAngle3, rotationVec3);

        Renderer::rasterizeModel(cube, screen, ppm);
        bMap = Renderer::getBrightnessMap(screen, ls);

        buf2.draw(bMap);
        buf2.activate();
        Renderer::clearScreen(screen);

        end = std::chrono::high_resolution_clock::now();

        elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::this_thread::sleep_for(std::chrono::milliseconds(std::max((long long)0, 30 - elapsed.count())));
    }
}