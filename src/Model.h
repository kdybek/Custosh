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
        Model(std::vector<Vector3<float>> vertices, std::vector<triangleIndices_t> triangles)
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

        void rotate(const Vector3<float>& origin,
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
        std::vector<Vector3<float>> m_vertices;
        std::vector<triangleIndices_t> m_triangles;

        static Vector3<float> rotatePoint(const Vector3<float>& origin,
                                          const Quaternion<float>& normalizedRotationQ,
                                          const Vector3<float>& p)
        {
            Vector3<float> originPVec3 = {p.x() - origin.x(), p.y() - origin.y(), p.z() - origin.z()};
            Quaternion<float> originPVec3AsQ(0.f, originPVec3);
            Quaternion<float> originNewVec3AsQ =
                    normalizedRotationQ * originPVec3AsQ * normalizedRotationQ.conjunction();

            return Vector3<float>(originNewVec3AsQ.getImaginaryVec() + origin);
        }

    }; // Model

} // Custosh


#endif // CUSTOSH_MODEL_H
