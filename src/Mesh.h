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
        Mesh(const std::vector<Vertex3D>& vertices, const std::vector<triangleIndices_t>& indices)
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

        void rotate(const Vertex3D& origin, const Vector3<float>& rotationVec, float angle)
        {
            Quaternion<float> rotationQ = {std::cos(angle / 2),
                                           Custosh::Vector3<float>(std::sin(angle / 2) * rotationVec)};

            Quaternion<float> normalizedQ = rotationQ * static_cast<float>(1.f / sqrt(rotationQ.normSq()));

            for (unsigned int i = 0; i < m_vertices.size(); ++i) {
                m_vertices.hostPtr()[i] = rotatePoint(origin, normalizedQ, m_vertices.hostPtr()[i]);
            }
        }

        [[nodiscard]] const HostDevPtr<Vertex3D>& hostDevVerticesPtr() const
        { return m_vertices; }

        [[nodiscard]] const HostDevPtr<triangleIndices_t>& hostDevIndicesPtr() const
        { return m_indices; }

    private:
        HostDevPtr<Vertex3D> m_vertices;
        HostDevPtr<triangleIndices_t> m_indices;

        static Vertex3D rotatePoint(const Vertex3D& origin,
                                    const Quaternion<float>& normalizedRotationQ,
                                    const Vertex3D& p)
        {
            Vertex3D originPVec3 = {p.x() - origin.x(), p.y() - origin.y(), p.z() - origin.z()};
            Quaternion<float> originPVec3AsQ(0.f, originPVec3);
            Quaternion<float> originNewVec3AsQ =
                    normalizedRotationQ * originPVec3AsQ * normalizedRotationQ.conjunction();

            return Vertex3D(originNewVec3AsQ.imaginaryVec() + origin);
        }

    }; // Mesh

} // Custosh


#endif // CUSTOSH_MESH_H
