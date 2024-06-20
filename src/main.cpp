#include <iostream>
#include <cmath>
#include <chrono>
#include <thread>

#include "Utility.h"
#include "Renderer.h"
#include "Model.h"

int main()
{
    std::vector<Custosh::Vector3<float>> cubeVer;
    cubeVer.push_back({20, 20, 110});
    cubeVer.push_back({60, 20, 110});
    cubeVer.push_back({20, 60, 110});
    cubeVer.push_back({60, 60, 110});

    cubeVer.push_back({20, 20, 150});
    cubeVer.push_back({60, 20, 150});
    cubeVer.push_back({20, 60, 150});
    cubeVer.push_back({60, 60, 150});

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

    Custosh::ResizableMatrix<Custosh::pixel_t> screen(100, 100);
    Custosh::PerspectiveMatrix pm(100, 1000);
    Custosh::lightSource_t ls = {.coords = {0, 0, 130}, .maxDistanceSq = 8000};

    float rotationAngle1 = Custosh::degreesToRadians(10);

    float rotationAngle2 = Custosh::degreesToRadians(4);

    float rotationAngle3 = Custosh::degreesToRadians(7);

    Custosh::Vector3<float> rotationVec1 = {0, 1, 0};

    Custosh::Vector3<float> rotationVec2 = {1, 0, 0};

    Custosh::Vector3<float> rotationVec3 = {0, 0, 1};

    while (true) {
        cube.rotate({40, 50, 130}, {std::cos(rotationAngle1 / 2),
                                       Custosh::Vector3<float>(std::sin(rotationAngle1 / 2) * rotationVec1)}, false);

        Custosh::Renderer::rasterizeModel(cube, screen, pm);
        Custosh::BrightnessMap bMap = Custosh::Renderer::getBrightnessMap(screen, ls);

        system("cls");
        std::cout << bMap;
        Custosh::Renderer::clearScreen(screen);
        std::this_thread::sleep_for(std::chrono::milliseconds(12));
    }
}