#ifndef CUSTOSH_RENDERER_H
#define CUSTOSH_RENDERER_H


#include "Utility.h"

namespace Custosh::Renderer
{
    // triangle3D vertices must have w = 1
    void rasterizeTriangle(const triangle3D_t& triangle3D,
                           ResizableMatrix<pixel_t>& screen,
                           const PerspectiveMatrix& pm);

    BrightnessMap getBrightnessMap(const ResizableMatrix<pixel_t>& screen,
                                   const lightSource_t& ls);

} // Custosh::Renderer


#endif // CUSTOSH_RENDERER_H
