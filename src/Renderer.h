#ifndef CUSTOSH_RENDERER_H
#define CUSTOSH_RENDERER_H


#include "Utility.h"

namespace Custosh::Renderer
{
    void rasterizeTriangle(const triangle_t& triangle2D, ResizableMatrix<float>& screen);

} // Custosh::Renderer


#endif // CUSTOSH_RENDERER_H
