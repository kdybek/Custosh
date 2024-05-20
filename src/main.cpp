#include <iostream>
#include <cmath>
#include <chrono>
#include <thread>

#include "Utility.h"
#include "Renderer.h"
#include "Model.h"

int main()
{
    std::vector<Custosh::Vector4<float>> cubeVer;
    cubeVer.push_back({20, 20, 110, 1});
    cubeVer.push_back({60, 20, 110, 1});
    cubeVer.push_back({20, 60, 110, 1});
    cubeVer.push_back({60, 60, 110, 1});

    cubeVer.push_back({20, 20, 150, 1});
    cubeVer.push_back({60, 20, 150, 1});
    cubeVer.push_back({20, 60, 150, 1});
    cubeVer.push_back({60, 60, 150, 1});

    std::vector<Custosh::triangleIndices_t> cubeInd;
    cubeInd.emplace_back(0, 1, 2);
    cubeInd.emplace_back(1, 2, 3);
    cubeInd.emplace_back(4, 5, 6);
    cubeInd.emplace_back(5, 6, 7);

    cubeInd.emplace_back(0, 4, 2);
    cubeInd.emplace_back(4, 6, 2);
    cubeInd.emplace_back(1, 5, 3);
    cubeInd.emplace_back(5, 7, 3);

    cubeInd.emplace_back(0, 1, 5);
    cubeInd.emplace_back(0, 4, 5);

    Custosh::Model cube(cubeVer, cubeInd);

    Custosh::ResizableMatrix<Custosh::pixel_t> screen(100, 100);
    Custosh::PerspectiveMatrix pm(100, 1000);
    Custosh::lightSource_t ls = {.coords = {40, 0, 90, 1}, .maxDistanceSq = 8000};

    float rotationAngle = Custosh::degreesToRadians(10);

    Custosh::Vector3<float> rotationVec3 = {0, 1, 0};

    while (true) {
        cube.rotate({40, 50, 130, 1}, {std::cos(rotationAngle / 2),
                                       Custosh::Vector3<float>(std::sin(rotationAngle / 2) * rotationVec3)}, false);

        Custosh::Renderer::rasterizeModel(cube, screen, pm);
        Custosh::BrightnessMap bMap = Custosh::Renderer::getBrightnessMap(screen, ls);

        system("cls");
        std::cout << bMap;
        Custosh::Renderer::clearScreen(screen);
        std::this_thread::sleep_for(std::chrono::milliseconds(12));
    }
}