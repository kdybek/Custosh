#ifndef CUSTOSH_RENDERER_H
#define CUSTOSH_RENDERER_H


#include "Utility.h"

namespace Custosh::Renderer
{
    void rasterizeTriangle(const triangle3D_t& triangle3D,
                           ResizableMatrix<int>& screen,
                           const PerspectiveMatrix& pm);

} // Custosh::Renderer


#endif // CUSTOSH_RENDERER_H
