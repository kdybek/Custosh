#ifndef CUSTOSH_RENDERER_CUH
#define CUSTOSH_RENDERER_CUH


#include "Mesh.h"

namespace Custosh::Renderer
{
        __host__ void drawMesh(const Mesh& mesh, const lightSource_t& ls);

} // Custosh::Renderer


#endif // CUSTOSH_RENDERER_CUH
