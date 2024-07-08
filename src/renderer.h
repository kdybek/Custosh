#ifndef CUSTOSH_RENDERER_H
#define CUSTOSH_RENDERER_H


#include "scene.h"

namespace custosh::renderer
{
    void loadScene(const Scene& scene);

    void loadTransformMatrix(const TransformMatrix& tm, unsigned int meshIdx);

    void draw();

} // custosh::renderer


#endif // CUSTOSH_RENDERER_H
