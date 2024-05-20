#ifndef CUSTOSH_MODEL_H
#define CUSTOSH_MODEL_H


#include <utility>
#include <vector>

#include "Utility.h"

namespace Custosh
{

    class Model
    {
    public:
        Model(std::vector<Vector4<float>> vertices, std::vector<triangleIndices_t> triangles)
                : m_vertices(std::move(vertices)),
                  m_triangles(std::move(triangles))
        {
        }

        [[nodiscard]] std::vector<triangle3D_t> getTriangles() const
        {
            std::vector<triangle3D_t> result;

            result.reserve(m_triangles.size());
            for (const triangleIndices_t& triangle: m_triangles) {
                result.emplace_back(m_vertices.at(triangle.p0),
                                    m_vertices.at(triangle.p1),
                                    m_vertices.at(triangle.p2));
            }

            return result;
        }

        // origin and p must have w = 1
        void rotate(const Vector4<float>& origin,
                    const Quaternion<float>& rotationQ,
                    bool normalizeQ)
        {
            Quaternion<float> normalizedQ = rotationQ;

            if (normalizeQ) { normalizedQ = rotationQ * static_cast<float>(1.f / sqrt(rotationQ.normSq())); }

            for (auto& v: m_vertices) {
                v = rotatePoint(origin, normalizedQ, v);
            }
        }

    private:
        std::vector<Vector4<float>> m_vertices;
        std::vector<triangleIndices_t> m_triangles;

        static Vector4<float> rotatePoint(const Vector4<float>& origin,
                                          const Quaternion<float>& normalizedRotationQ,
                                          const Vector4<float>& p)
        {
            Vector3<float> originPVec3 = {p.x() - origin.x(), p.y() - origin.y(), p.z() - origin.z()};
            Quaternion<float> originPVec3AsQ(0.f, originPVec3);
            Quaternion<float> originNewVec3AsQ =
                    normalizedRotationQ * originPVec3AsQ * normalizedRotationQ.conjunction();
            Vector3<float> originNewVec3 = originNewVec3AsQ.getImaginaryVec();

            return {originNewVec3.x() + origin.x(),
                    originNewVec3.y() + origin.y(),
                    originNewVec3.z() + origin.z(),
                    1.f};
        }

    }; // Model

} // Custosh


#endif // CUSTOSH_MODEL_H
