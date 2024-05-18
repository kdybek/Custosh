#include "Renderer.h"

#include <cmath>

#include "CustoshExcept.h"

namespace Custosh::Renderer
{
    namespace
    {
        boundingBox_t findBounds(const triangle_t& triangle2D, const ResizableMatrix<pixel_t>& screen)
        {
            boundingBox_t boundingBox{};
            boundingBox.xMax = std::max({static_cast<int>(std::ceil(triangle2D.p0.x())),
                                         static_cast<int>(std::ceil(triangle2D.p1.x())),
                                         static_cast<int>(std::ceil(triangle2D.p2.x())),
                                         static_cast<int>(screen.getNCols() - 1)});
            boundingBox.xMin = std::min({static_cast<int>(std::floor(triangle2D.p0.x())),
                                         static_cast<int>(std::floor(triangle2D.p1.x())),
                                         static_cast<int>(std::floor(triangle2D.p2.x())),
                                         0});
            boundingBox.yMax = std::max({static_cast<int>(std::ceil(triangle2D.p0.y())),
                                         static_cast<int>(std::ceil(triangle2D.p1.y())),
                                         static_cast<int>(std::ceil(triangle2D.p2.y())),
                                         static_cast<int>(screen.getNRows()) - 1});
            boundingBox.yMax = std::min({static_cast<int>(std::floor(triangle2D.p0.y())),
                                         static_cast<int>(std::floor(triangle2D.p1.y())),
                                         static_cast<int>(std::floor(triangle2D.p2.y())),
                                         0});
            return boundingBox;
        }

    } // anonymous

    void rasterizeTriangle(const triangle_t& triangle2D, ResizableMatrix<pixel_t>& screen)
    {
        if (triangle2D.p0.z() != 0 || triangle2D.p1.z() != 0 || triangle2D.p2.z() != 0) {
            throw CustoshException("triangle to be rasterized is not projected to the screen");
        }

        if (screen.getNRows() == 0 || screen.getNCols() == 0) {
            return;
        }

        boundingBox_t boundingBox = findBounds(triangle2D, screen);


    }

} // Custosh::Renderer
