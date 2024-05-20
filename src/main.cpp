#include <iostream>
#include <cmath>

#include "Utility.h"
#include "Renderer.h"

int main()
{
    Custosh::triangle3D_t triangle3D{};
    triangle3D.p0 = {1, 2, 3, 1};
    triangle3D.p1 = {15, 15, 5, 1};
    triangle3D.p2 = {1, 15, 3, 1};
    Custosh::ResizableMatrix<Custosh::pixel1_t> resizableMatrix(20, 20);
    Custosh::PerspectiveMatrix pm(3, 7);
    Custosh::Renderer::rasterizeTriangle(triangle3D, resizableMatrix, pm);
    std::cout << resizableMatrix(9, 9).coords;
}