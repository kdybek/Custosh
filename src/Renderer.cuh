#ifndef CUSTOSH_RENDERER_CUH
#define CUSTOSH_RENDERER_CUH


#include "Utility.cuh"
#include "Mesh.h"

namespace Custosh::Renderer
{
    void clearScreen(HostDevResizableMatrix<pixel_t>& screen);

    void rasterizeModel(const Mesh& mesh,
                        HostDevResizableMatrix<pixel_t>& screen,
                        const PerspectiveProjMatrix& ppm);

    void rasterizeTriangle(triangle3D_t triangle3D,
                           HostDevResizableMatrix<pixel_t>& screen,
                           const PerspectiveProjMatrix& ppm);

    BrightnessMap getBrightnessMap(const HostDevResizableMatrix<pixel_t>& screen,
                                   const lightSource_t& ls);

} // Custosh::Renderer


#endif // CUSTOSH_RENDERER_CUH
