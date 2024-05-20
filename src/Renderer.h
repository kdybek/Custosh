#ifndef CUSTOSH_RENDERER_H
#define CUSTOSH_RENDERER_H


#include "Utility.h"

namespace Custosh::Renderer
{
    // triangle3D vertices must have w = 1
    void rasterizeTriangle(const triangle3D_t& triangle3D,
                           ResizableMatrix<pixel1_t>& screen,
                           const PerspectiveMatrix& pm);

} // Custosh::Renderer


#endif // CUSTOSH_RENDERER_H
