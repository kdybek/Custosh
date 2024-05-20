#include "Renderer.h"

#include <cmath>

#include "CustoshExcept.h"

namespace Custosh::Renderer
{
    namespace
    {
        template <typename T>
        boundingBox_t findBounds(const triangle2D_t& triangle2D,
                                 const ResizableMatrix<T>& screen)
        {
            boundingBox_t boundingBox{};
            float xMax = std::max({std::ceil(triangle2D.p0.x()),
                                   std::ceil(triangle2D.p1.x()),
                                   std::ceil(triangle2D.p2.x())});
            float xMin = std::min({std::floor(triangle2D.p0.x()),
                                   std::floor(triangle2D.p1.x()),
                                   std::floor(triangle2D.p2.x())});
            float yMax = std::max({std::ceil(triangle2D.p0.y()),
                                   std::ceil(triangle2D.p1.y()),
                                   std::ceil(triangle2D.p2.y())});
            float yMin = std::min({std::floor(triangle2D.p0.y()),
                                   std::floor(triangle2D.p1.y()),
                                   std::floor(triangle2D.p2.y())});

            boundingBox.xMax = std::min(static_cast<int>(xMax), static_cast<int>(screen.getNCols()));
            boundingBox.xMin = std::min(static_cast<int>(xMin), 0);
            boundingBox.yMax = std::min(static_cast<int>(yMax), static_cast<int>(screen.getNRows()));
            boundingBox.yMin = std::min(static_cast<int>(yMin), 0);
            return boundingBox;
        }

        bool isBottomOrRight(const Vector2<float>& a,
                             const Vector2<float>& b)
        {
            Vector2<float> edge = {b.x() - a.x(), b.y() - a.y()};
            bool bottomEdge = edge.y() == 0 && edge.x() < 0;
            bool rightEdge = edge.y() < 0;

            return bottomEdge || rightEdge;
        }

        float cross2D(const Vector2<float>& a,
                      const Vector2<float>& b,
                      const Vector2<float>& c)
        {
            Vector2<float> ab = {b.x() - a.x(), b.y() - a.y()};
            Vector2<float> ac = {c.x() - a.x(), c.y() - a.y()};
            return ab.x() * ac.y() - ab.y() * ac.x();
        }

        bool inTriangle(const triangle2D_t& triangle2D,
                        const Vector2<float>& p,
                        float triangleArea,
                        barycentricCoords_t& barycentricCoords)
        {
            float w0 = cross2D(triangle2D.p1, triangle2D.p2, p);
            float w1 = cross2D(triangle2D.p2, triangle2D.p0, p);
            float w2 = cross2D(triangle2D.p0, triangle2D.p1, p);

            if (w0 == 0 && isBottomOrRight(triangle2D.p1, triangle2D.p2)) { return false; }
            if (w1 == 0 && isBottomOrRight(triangle2D.p2, triangle2D.p0)) { return false; }
            if (w2 == 0 && isBottomOrRight(triangle2D.p0, triangle2D.p1)) { return false; }

            barycentricCoords.alpha = w0 / triangleArea;
            barycentricCoords.beta = w1 / triangleArea;
            barycentricCoords.gamma = w2 / triangleArea;

            return (w0 >= 0.f && w1 >= 0.f && w2 >= 0.f);
        }

        Vector2<float> applyPerspectivePoint(const Vector4<float>& p,
                                             const PerspectiveMatrix& pm)
        {
            Vector4<float> pPerspective = Vector4(pm * p).normalizeW();
            return {pPerspective.x(), pPerspective.y()};
        }

        triangle2D_t applyPerspectiveTriangle(const triangle3D_t& triangle3D,
                                              const PerspectiveMatrix& pm)
        {
            return {.p0 = applyPerspectivePoint(triangle3D.p0, pm),
                    .p1 = applyPerspectivePoint(triangle3D.p1, pm),
                    .p2 = applyPerspectivePoint(triangle3D.p2, pm)};
        }

        Vector4<float> getCartesianCoords(const triangle3D_t& triangle3D, const barycentricCoords_t& bc)
        {
            return {triangle3D.p0.x() * bc.alpha + triangle3D.p1.x() * bc.beta + triangle3D.p2.x() * bc.gamma,
                    triangle3D.p0.y() * bc.alpha + triangle3D.p1.y() * bc.beta + triangle3D.p2.y() * bc.gamma,
                    triangle3D.p0.z() * bc.alpha + triangle3D.p1.z() * bc.beta + triangle3D.p2.z() * bc.gamma,
                    1.f};
        }

    } // anonymous

    void rasterizeTriangle(const triangle3D_t& triangle3D,
                           ResizableMatrix<pixel1_t>& screen,
                           const PerspectiveMatrix& pm)
    {
        triangle2D_t triangle2D = applyPerspectiveTriangle(triangle3D, pm);

        float triangleArea = cross2D(triangle2D.p0, triangle2D.p1, triangle2D.p2);
        barycentricCoords_t bc{};

        if (screen.getNRows() == 0 || screen.getNCols() == 0 || triangleArea == 0.f) {
            return;
        }

        boundingBox_t boundingBox = findBounds(triangle2D, screen);

        for (int i = boundingBox.xMin; i <= boundingBox.xMax; ++i) {
            for (int j = boundingBox.yMin; j <= boundingBox.yMax; ++j) {

                if (inTriangle(triangle2D,
                               Vector2<float>({static_cast<float>(i), static_cast<float>(j)}),
                               triangleArea,
                               bc)) {
                    Vector4<float> projectedPoint = getCartesianCoords(triangle3D, bc);
                    pixel1_t& screenPoint = screen(j, i);

                    if (!screenPoint.occupied || screenPoint.coords.z() < projectedPoint.z()) {
                        screenPoint.occupied = true;
                        screenPoint.coords = projectedPoint;
                    }
                }
            }
        }

    }

} // Custosh::Renderer
