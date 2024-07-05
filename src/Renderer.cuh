#ifndef CUSTOSH_RENDERER_CUH
#define CUSTOSH_RENDERER_CUH


#include "Scene.h"

namespace Custosh::Renderer
{
    __host__ void loadScene(const Scene& scene);

    __host__ void loadTransformMat(const TransformMatrix& tm);

    __host__ void draw(const lightSource_t& ls);

} // Custosh::Renderer


#endif // CUSTOSH_RENDERER_CUH
