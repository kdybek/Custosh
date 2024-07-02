#ifndef CUSTOSH_MESH_H
#define CUSTOSH_MESH_H


#include <utility>
#include <vector>

#include "Utility.cuh"

namespace Custosh
{
    class Mesh
    {
    public:
        Mesh(const std::vector<Vector3<float>>& vertices, const std::vector<triangleIndices_t>& indices)
                : m_vertices(vertices.size()),
                  m_indices(indices.size())
        {
            for (unsigned int i = 0; i < vertices.size(); ++i) {
                m_vertices.hostPtr()[i] = vertices[i];
            }

            for (unsigned int i = 0; i < indices.size(); ++i) {
                m_indices.hostPtr()[i] = indices[i];
            }
        }

        void rotate(const Vector3<float>& origin, const Vector3<float>& rotationVec, float angle)
        {
            Quaternion<float> rotationQ = {std::cos(angle / 2),
                                           Custosh::Vector3<float>(std::sin(angle / 2) * rotationVec)};

            Quaternion<float> normalizedQ = rotationQ * static_cast<float>(1.f / sqrt(rotationQ.normSq()));

            for (unsigned int i = 0; i < m_vertices.size(); ++i) {
                m_vertices.hostPtr()[i] = rotatePoint(origin, normalizedQ, m_vertices.hostPtr()[i]);
            }
        }

        [[nodiscard]] const HostDevPtr<Vector3<float>>& hostDevVerticesPtr() const
        { return m_vertices; }

        [[nodiscard]] const HostDevPtr<triangleIndices_t>& hostDevIndicesPtr() const
        { return m_indices; }

    private:
        HostDevPtr<Vector3<float>> m_vertices;
        HostDevPtr<triangleIndices_t> m_indices;

        static Vector3<float> rotatePoint(const Vector3<float>& origin,
                                          const Quaternion<float>& normalizedRotationQ,
                                          const Vector3<float>& p)
        {
            Vector3<float> originPVec3 = {p.x() - origin.x(), p.y() - origin.y(), p.z() - origin.z()};
            Quaternion<float> originPVec3AsQ(0.f, originPVec3);
            Quaternion<float> originNewVec3AsQ =
                    normalizedRotationQ * originPVec3AsQ * normalizedRotationQ.conjunction();

            return Vector3<float>(originNewVec3AsQ.imaginaryVec() + origin);
        }

    }; // Mesh

} // Custosh


#endif // CUSTOSH_MESH_H
