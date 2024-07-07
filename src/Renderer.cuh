#ifndef CUSTOSH_RENDERER_CUH
#define CUSTOSH_RENDERER_CUH


#include "Scene.h"

namespace Custosh::Renderer
{
    __host__ void loadScene(const Scene& scene);

    __host__ void loadTransformMatrix(const TransformMatrix& tm, unsigned int meshIdx);

    __host__ void setLightSource(const lightSource_t& ls);

    __host__ void draw();

} // Custosh::Renderer


#endif // CUSTOSH_RENDERER_CUH
