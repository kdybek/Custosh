#include "Renderer.h"

#include <cmath>
#include <algorithm>

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
                        float triangleArea2x,
                        barycentricCoords_t& barycentricCoords)
        {
            float w0 = cross2D(triangle2D.p1, p, triangle2D.p2);
            float w1 = cross2D(triangle2D.p2, p, triangle2D.p0);
            float w2 = cross2D(triangle2D.p0, p, triangle2D.p1);

            if (w0 == 0 && isBottomOrRight(triangle2D.p1, triangle2D.p2)) { return false; }
            if (w1 == 0 && isBottomOrRight(triangle2D.p2, triangle2D.p0)) { return false; }
            if (w2 == 0 && isBottomOrRight(triangle2D.p0, triangle2D.p1)) { return false; }

            barycentricCoords.alpha = w0 / triangleArea2x;
            barycentricCoords.beta = w1 / triangleArea2x;
            barycentricCoords.gamma = w2 / triangleArea2x;

            return (w0 >= 0.f && w1 >= 0.f && w2 >= 0.f);
        }

        Vector2<float> applyPerspectivePoint(const Vector3<float>& p,
                                             const PPM& ppm)
        {
            Vector4<float> pPerspective = Vector4<float>(ppm * p.toHomogeneous()).normalizeW();
            return {pPerspective.x(), pPerspective.y()};
        }

        triangle2D_t applyPerspectiveTriangle(const triangle3D_t& triangle3D,
                                              const PPM& ppm)
        {
            return {.p0 = applyPerspectivePoint(triangle3D.p0, ppm),
                    .p1 = applyPerspectivePoint(triangle3D.p1, ppm),
                    .p2 = applyPerspectivePoint(triangle3D.p2, ppm)};
        }

        Vector3<float> getCartesianCoords(const triangle3D_t& triangle3D, const barycentricCoords_t& bc)
        {
            return {triangle3D.p0.x() * bc.alpha + triangle3D.p1.x() * bc.beta + triangle3D.p2.x() * bc.gamma,
                    triangle3D.p0.y() * bc.alpha + triangle3D.p1.y() * bc.beta + triangle3D.p2.y() * bc.gamma,
                    triangle3D.p0.z() * bc.alpha + triangle3D.p1.z() * bc.beta + triangle3D.p2.z() * bc.gamma};
        }

        float distanceSq(const Vector3<float>& a, const Vector3<float>& b)
        {
            return static_cast<float>(pow((a.x() - b.x()), 2) + pow((a.y() - b.y()), 2) + pow((a.z() - b.z()), 2));
        }

        float cosine3D(const Vector3<float>& center, const Vector3<float>& p1, const Vector3<float>& p2)
        {
            auto vec1 = Vector3<float>(p1 - center);
            auto vec2 = Vector3<float>(p2 - center);
            float dist1 = std::sqrt(distanceSq(center, p1));
            float dist2 = std::sqrt(distanceSq(center, p2));

            return vec1.dot(vec2) / (dist1 * dist2);
        }

        float getPointBrightness(const pixel_t& p, const lightSource_t& ls)
        {
            float distSq = distanceSq(p.coords, ls.coords);
            float cos = cosine3D(p.coords, Vector3<float>(p.coords + p.normal), ls.coords);

            return std::clamp(std::max(cos, 0.f) * ls.intensity / distSq, 0.f, 1.f);
        }

        // The vertices are clockwise oriented, but we're looking from 0 towards positive z values.
        Vector3<float> triangleNormal(const triangle3D_t& triangle3D)
        {
            Vector3<float> res = Vector3<float>(triangle3D.p1 - triangle3D.p0).cross(
                    Vector3<float>(triangle3D.p2 - triangle3D.p0));

            return Vector3<float>(res.normalized());
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
                        const PPM& ppm)
    {
        for (const auto& triangle: model.getTriangles()) {
            rasterizeTriangle(triangle, screen, ppm);
        }
    }

    void rasterizeTriangle(triangle3D_t triangle3D,
                           ResizableMatrix<pixel_t>& screen,
                           const PPM& ppm)
    {
        triangle2D_t triangle2D = applyPerspectiveTriangle(triangle3D, ppm);
        float triangleArea2x = cross2D(triangle2D.p0, triangle2D.p2, triangle2D.p1);
        barycentricCoords_t bc{};

        // In other functions the triangles' vertices are assumed to be in a clockwise order.
        if (triangleArea2x < 0.f) {
            std::swap(triangle2D.p0, triangle2D.p1);
            std::swap(triangle3D.p0, triangle3D.p1);
            triangleArea2x *= -1;
        }

        Vector3<float> normal = triangleNormal(triangle3D);

        if (screen.getNRows() == 0 || screen.getNCols() == 0 || triangleArea2x == 0.f) {
            return;
        }

        boundingBox_t boundingBox = findBounds(triangle2D, screen);

        for (int i = boundingBox.yMin; i <= boundingBox.yMax; ++i) {
            for (int j = boundingBox.xMin; j <= boundingBox.xMax; ++j) {

                if (inTriangle(triangle2D,
                               Vector2<float>({static_cast<float>(j), static_cast<float>(i)}),
                               triangleArea2x,
                               bc)) {
                    Vector3<float> projectedPoint = getCartesianCoords(triangle3D, bc);
                    pixel_t& screenPoint = screen(i, j);

                    if (!screenPoint.occupied || screenPoint.coords.z() > projectedPoint.z()) {
                        screenPoint.occupied = true;
                        screenPoint.coords = projectedPoint;
                        screenPoint.normal = normal;
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
                    bMap(i, j) = getPointBrightness(screenPoint, ls);
                }
            }
        }

        return bMap;
    }

} // Custosh::Renderer
