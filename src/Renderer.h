#ifndef CUSTOSH_RENDERER_H
#define CUSTOSH_RENDERER_H


#include "Utility.h"
#include "Model.h"

namespace Custosh::Renderer
{
    void clearScreen(ResizableMatrix<pixel_t>& screen);

    void rasterizeModel(const Model& model,
                        ResizableMatrix<pixel_t>& screen,
                        const PerspectiveMatrix& pm);

    void rasterizeTriangle(triangle3D_t triangle3D,
                           ResizableMatrix<pixel_t>& screen,
                           const PerspectiveMatrix& pm);

    BrightnessMap getBrightnessMap(const ResizableMatrix<pixel_t>& screen,
                                   const lightSource_t& ls);

} // Custosh::Renderer


#endif // CUSTOSH_RENDERER_H
