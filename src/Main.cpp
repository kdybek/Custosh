#include <iostream>
#include <cmath>
#include <chrono>
#include <thread>

#include "Utility.h"
#include "Renderer.h"
#include "Model.h"
#include "WindowsConsoleScreenBuffer.h"

int main()
{
    std::vector<Custosh::Vector3<float>> cubeVer;
    cubeVer.push_back({-0.5, -0.5, 1.5});
    cubeVer.push_back({0.5, -0.5, 1.5});
    cubeVer.push_back({-0.5, 0.5, 1.5});
    cubeVer.push_back({0.5, 0.5, 1.5});

    cubeVer.push_back({-0.5, -0.5, 2.5});
    cubeVer.push_back({0.5, -0.5, 2.5});
    cubeVer.push_back({-0.5, 0.5, 2.5});
    cubeVer.push_back({0.5, 0.5, 2.5});

    std::vector<Custosh::triangleIndices_t> cubeInd;
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

    Custosh::Model cube(cubeVer, cubeInd);

    Custosh::ResizableMatrix<Custosh::pixel_t> screen(70, 70);
    Custosh::OrtProjMatrix opm({-1, -1, 1},
                               {1, 1, 10},
                               {0, 0, 0},
                               {70, 70, 0});

    Custosh::PerspectiveMatrix pm(1, 10);
    Custosh::PPM ppm(pm, opm);

    Custosh::lightSource_t ls = {.coords = {-5, 5, 2}, .maxDistanceSq = 70};

    float rotationAngle1 = Custosh::degreesToRadians(10);

    float rotationAngle2 = Custosh::degreesToRadians(4);

    float rotationAngle3 = Custosh::degreesToRadians(7);

    Custosh::Vector3<float> rotationVec1 = {0, 1, 0};

    Custosh::Vector3<float> rotationVec2 = {1, 0, 0};

    Custosh::Vector3<float> rotationVec3 = {0, 0, 1};

    Custosh::WindowsConsoleScreenBuffer buf1;
    Custosh::WindowsConsoleScreenBuffer buf2;

    while (true) {
        cube.rotate({0, 0, 2}, {std::cos(rotationAngle1 / 2),
                                Custosh::Vector3<float>(std::sin(rotationAngle1 / 2) * rotationVec1)}, false);
        cube.rotate({0, 0, 2}, {std::cos(rotationAngle2 / 2),
                                Custosh::Vector3<float>(std::sin(rotationAngle2 / 2) * rotationVec2)}, false);
        cube.rotate({0, 0, 2}, {std::cos(rotationAngle3 / 2),
                                Custosh::Vector3<float>(std::sin(rotationAngle3 / 2) * rotationVec3)}, false);

        Custosh::Renderer::rasterizeModel(cube, screen, ppm);
        Custosh::BrightnessMap bMap = Custosh::Renderer::getBrightnessMap(screen, ls);

        buf1.draw(bMap);
        buf1.activate();
        Custosh::Renderer::clearScreen(screen);

        cube.rotate({0, 0, 2}, {std::cos(rotationAngle1 / 2),
                                Custosh::Vector3<float>(std::sin(rotationAngle1 / 2) * rotationVec1)}, false);
        cube.rotate({0, 0, 2}, {std::cos(rotationAngle2 / 2),
                                Custosh::Vector3<float>(std::sin(rotationAngle2 / 2) * rotationVec2)}, false);
        cube.rotate({0, 0, 2}, {std::cos(rotationAngle3 / 2),
                                Custosh::Vector3<float>(std::sin(rotationAngle3 / 2) * rotationVec3)}, false);

        Custosh::Renderer::rasterizeModel(cube, screen, ppm);
        bMap = Custosh::Renderer::getBrightnessMap(screen, ls);

        buf2.draw(bMap);
        buf2.activate();
        Custosh::Renderer::clearScreen(screen);
    }
}