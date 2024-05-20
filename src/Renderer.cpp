#include "Renderer.h"

#include <cmath>

namespace Custosh::Renderer
{
    namespace
    {
        template<typename T>
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

            boundingBox.xMax = std::min(static_cast<int>(xMax), static_cast<int>(screen.getNCols() - 1));
            boundingBox.xMin = std::max(static_cast<int>(xMin), 0);
            boundingBox.yMax = std::min(static_cast<int>(yMax), static_cast<int>(screen.getNRows() - 1));
            boundingBox.yMin = std::max(static_cast<int>(yMin), 0);
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
                        float triangleAreaTimeTwo,
                        barycentricCoords_t& barycentricCoords)
        {
            float w0 = cross2D(triangle2D.p1, triangle2D.p2, p);
            float w1 = cross2D(triangle2D.p2, triangle2D.p0, p);
            float w2 = cross2D(triangle2D.p0, triangle2D.p1, p);

            if (w0 == 0 && isBottomOrRight(triangle2D.p1, triangle2D.p2)) { return false; }
            if (w1 == 0 && isBottomOrRight(triangle2D.p2, triangle2D.p0)) { return false; }
            if (w2 == 0 && isBottomOrRight(triangle2D.p0, triangle2D.p1)) { return false; }

            barycentricCoords.alpha = w0 / triangleAreaTimeTwo;
            barycentricCoords.beta = w1 / triangleAreaTimeTwo;
            barycentricCoords.gamma = w2 / triangleAreaTimeTwo;

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

        Vector4<float> getCartesianCoords(const triangle3D_t& triangle3D, const barycentricCoords_t& bc, bool swap)
        {
            if (!swap) {
                return {triangle3D.p0.x() * bc.alpha + triangle3D.p1.x() * bc.beta + triangle3D.p2.x() * bc.gamma,
                        triangle3D.p0.y() * bc.alpha + triangle3D.p1.y() * bc.beta + triangle3D.p2.y() * bc.gamma,
                        triangle3D.p0.z() * bc.alpha + triangle3D.p1.z() * bc.beta + triangle3D.p2.z() * bc.gamma,
                        1.f};
            }
            else {
                return {triangle3D.p1.x() * bc.alpha + triangle3D.p0.x() * bc.beta + triangle3D.p2.x() * bc.gamma,
                        triangle3D.p1.y() * bc.alpha + triangle3D.p0.y() * bc.beta + triangle3D.p2.y() * bc.gamma,
                        triangle3D.p1.z() * bc.alpha + triangle3D.p0.z() * bc.beta + triangle3D.p2.z() * bc.gamma,
                        1.f};
            }
        }

        float distanceSq(const Vector4<float>& a, const Vector4<float>& b)
        {
            return static_cast<float>(pow((a.x() - b.x()), 2) + pow((a.y() - b.y()), 2) + pow((a.z() - b.z()), 2));
        }

        float getPointBrightness(const Vector4<float>& p, const lightSource_t& ls)
        {
            float distSq = distanceSq(p, ls.coords);
            float brightness = 1 - distSq / ls.maxDistanceSq;
            if (brightness > 0) { return brightness; }
            else { return 0; }
        }

    } // anonymous

    void clearScreen(ResizableMatrix<pixel_t>& screen)
    {
        for (unsigned int i = 0; i < screen.getNRows(); ++i) {
            for (unsigned int j = 0; j < screen.getNCols(); ++j) {
                screen(i, j).occupied = false;
            }
        }
    }

    void rasterizeModel(const Model& model,
                        ResizableMatrix<pixel_t>& screen,
                        const PerspectiveMatrix& pm)
    {
        for (const auto& triangle: model.getTriangles()) {
            rasterizeTriangle(triangle, screen, pm);
        }
    }

    void rasterizeTriangle(const triangle3D_t& triangle3D,
                           ResizableMatrix<pixel_t>& screen,
                           const PerspectiveMatrix& pm)
    {
        triangle2D_t triangle2D = applyPerspectiveTriangle(triangle3D, pm);
        float triangleAreaTimesTwo = cross2D(triangle2D.p0, triangle2D.p1, triangle2D.p2);
        barycentricCoords_t bc{};
        bool swap = false;

        if (triangleAreaTimesTwo < 0.f) {
            std::swap(triangle2D.p0, triangle2D.p1);
            triangleAreaTimesTwo *= -1;
            swap = true;
        }

        if (screen.getNRows() == 0 || screen.getNCols() == 0 || triangleAreaTimesTwo == 0.f) {
            return;
        }

        boundingBox_t boundingBox = findBounds(triangle2D, screen);

        for (int i = boundingBox.yMin; i <= boundingBox.yMax; ++i) {
            for (int j = boundingBox.xMin; j <= boundingBox.xMax; ++j) {

                if (inTriangle(triangle2D,
                               Vector2<float>({static_cast<float>(j), static_cast<float>(i)}),
                               triangleAreaTimesTwo,
                               bc)) {
                    Vector4<float> projectedPoint = getCartesianCoords(triangle3D, bc, swap);
                    pixel_t& screenPoint = screen(i, j);

                    if (!screenPoint.occupied || screenPoint.coords.z() < projectedPoint.z()) {
                        screenPoint.occupied = true;
                        screenPoint.coords = projectedPoint;
                    }
                }
            }
        }
    }

    BrightnessMap getBrightnessMap(const ResizableMatrix<pixel_t>& screen,
                                            const lightSource_t& ls)
    {
        BrightnessMap bMap(screen.getNRows(), screen.getNCols());

        for (unsigned int i = 0; i < screen.getNRows(); ++i) {
            for (unsigned int j = 0; j < screen.getNCols(); ++j) {
                const pixel_t& screenPoint = screen(i, j);

                if (screenPoint.occupied) {
                    bMap(i, j) = getPointBrightness(screenPoint.coords, ls);
                }
            }
        }

        return bMap;
    }

} // Custosh::Renderer
