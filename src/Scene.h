#ifndef CUSTOSH_SCENE_H
#define CUSTOSH_SCENE_H


#include "Utility.cuh"

namespace Custosh
{
    class Mesh
    {
    public:
        Mesh(std::vector<Vertex3D> vertices, std::vector<triangleIndices_t> indices)
                : m_vertices(std::move(vertices)), m_triangles(std::move(indices))
        {
        }

        [[nodiscard]] inline const std::vector<Vertex3D>& vertices() const
        { return m_vertices; }

        [[nodiscard]] inline const std::vector<triangleIndices_t>& triangles() const
        { return m_triangles; }

        void rotate(const Vertex3D& origin, const Vector3<float>& rotationVec, float angle)
        {
            Quaternion<float> rotationQ = {std::cos(angle / 2),
                                           Custosh::Vector3<float>(std::sin(angle / 2) * rotationVec)};

            Quaternion<float> normalizedQ = rotationQ * static_cast<float>(1.f / sqrt(rotationQ.normSq()));

            for (auto & vertex : m_vertices) {
                vertex = rotatePoint(origin, normalizedQ, vertex);
            }
        }

    private:
        std::vector<Vertex3D> m_vertices;
        std::vector<triangleIndices_t> m_triangles;

        static Vertex3D rotatePoint(const Vertex3D& origin,
                                    const Quaternion<float>& normalizedRotationQ,
                                    const Vertex3D& p)
        {
            auto centerTranslationVec = Vector3<float>(Vertex3D({0.f, 0.f, 0.f}) - origin);
            TranslationMatrix centerTranslationMat(centerTranslationVec);
            RotationMatrix rotationMat(normalizedRotationQ);
            TranslationMatrix originTranslationMat(origin);

            Vector4<float> rotatedP = Vector4<float>(originTranslationMat * rotationMat * centerTranslationMat * p.toHomogeneous());

            return {rotatedP.x(), rotatedP.y(), rotatedP.z()};
        }

    }; // Mesh

    class Scene
    {
    public:
        Scene() : m_vertices(0), m_triangles(0)
        {
        }

        void add(const Mesh& mesh)
        {
            const unsigned int firstVertexIdx = m_vertices.size();
            const unsigned int firstTriangleIdx = m_triangles.size();

            m_vertices.resizeAndCopy(m_vertices.size() + mesh.vertices().size());
            m_triangles.resizeAndCopy(m_triangles.size() + mesh.triangles().size());

            for (unsigned int i = firstVertexIdx; i < m_vertices.size(); ++i) {
                m_vertices.get()[i] = mesh.vertices()[i - firstVertexIdx];
            }

            for (unsigned int i = firstTriangleIdx; i < m_triangles.size(); ++i) {
                m_triangles.get()[i] = offsetTriangleIndices(mesh.triangles()[i - firstVertexIdx], firstTriangleIdx);
            }
        }

        [[nodiscard]] const HostPtr<Vertex3D>& verticesPtr() const
        { return m_vertices; }

        [[nodiscard]] const HostPtr<triangleIndices_t>& indicesPtr() const
        { return m_triangles; }

    private:
        HostPtr<Vertex3D> m_vertices;
        HostPtr<triangleIndices_t> m_triangles;

        static triangleIndices_t offsetTriangleIndices(const triangleIndices_t& ti, unsigned int offset)
        {
            return triangleIndices_t(ti.p0 + offset, ti.p1 + offset, ti.p2 + offset);
        }

    }; // Scene

} // Custosh


#endif // CUSTOSH_SCENE_H
