#ifndef CUSTOSH_RENDERER_H
#define CUSTOSH_RENDERER_H


#include "scene.h"

namespace Custosh::Renderer
{
    void setScene(const Scene& scene);

    void setTransformationMatrix(const TransformationMatrix& tm,
                                 unsigned int meshIdx);

    void transformVerticesAndDraw();

} // Custosh::Renderer


#endif // CUSTOSH_RENDERER_H
