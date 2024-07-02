#ifndef CUSTOSH_RENDERER_CUH
#define CUSTOSH_RENDERER_CUH


#include "Mesh.h"

namespace Custosh::Renderer
{
        __host__ void drawMesh(const Mesh& mesh, const PerspectiveProjMatrix& ppm, const lightSource_t& ls);

        // TODO: void set rows and cols function

} // Custosh::Renderer


#endif // CUSTOSH_RENDERER_CUH
