#ifndef CUSTOSH_RENDERER_H
#define CUSTOSH_RENDERER_H


#include "scene.h"

namespace Custosh::Renderer
{
    void loadScene(const Scene& scene);

    void loadTransformMatrix(const TransformMatrix& tm, unsigned int meshIdx);

    void draw();

} // Custosh::Renderer


#endif // CUSTOSH_RENDERER_H
