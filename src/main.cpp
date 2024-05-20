#include <iostream>
#include <cmath>
#include <chrono>
#include <thread>

#include "Utility.h"
#include "Renderer.h"

int main()
{
    Custosh::triangle3D_t triangle3D{};
    triangle3D.p0 = {50, 0, 110, 1};
    triangle3D.p1 = {80, 90, 110, 1};
    triangle3D.p2 = {0, 90, 110, 1};

    Custosh::ResizableMatrix<Custosh::pixel_t> screen(100, 100);
    Custosh::PerspectiveMatrix pm(100, 1000);
    Custosh::lightSource_t ls = {.coords = {40, 0, 2, 1}, .maxDistanceSq = 20000};

    float rotationAngle = Custosh::degreesToRadians(10);

    Custosh::Vector3<float> rotationVec3 = {0, 1, 0};

    while (true) {

        triangle3D.p2 = Custosh::rotatePoint({40, 50, 110, 1},
                                             {std::cos(rotationAngle / 2),
                                              Custosh::Vector3<float>(std::sin(rotationAngle / 2) * rotationVec3)},
                                             triangle3D.p2,
                                             false);

        triangle3D.p1 = Custosh::rotatePoint({40, 50, 110, 1},
                                             {std::cos(rotationAngle / 2),
                                              Custosh::Vector3<float>(std::sin(rotationAngle / 2) * rotationVec3)},
                                             triangle3D.p1,
                                             false);


        Custosh::Renderer::rasterizeTriangle(triangle3D, screen, pm);
        Custosh::BrightnessMap bMap = Custosh::Renderer::getBrightnessMap(screen, ls);

        system("cls");
        std::cout << bMap;
        Custosh::clearScreen(screen);
        std::this_thread::sleep_for(std::chrono::milliseconds(12));
    }
}