#ifndef CUSTOSH_SCENE_H
#define CUSTOSH_SCENE_H


#include "utility.cuh"

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

    private:
        std::vector<Vertex3D> m_vertices;
        std::vector<triangleIndices_t> m_triangles;

    }; // Mesh

    class Scene
    {
    public:
        Scene() : m_vertices(0), m_triangles(0), m_firstVertexIdxPerMesh(), m_numVerticesPerMesh()
        {
        }

        // Returns the mesh index withing the scene.
        unsigned int add(const Mesh& mesh)
        {
            const unsigned int addedMeshIdx = m_firstVertexIdxPerMesh.size();
            const unsigned int firstVertexIdx = m_vertices.size();
            const unsigned int firstTriangleIdx = m_triangles.size();

            m_firstVertexIdxPerMesh.push_back(firstVertexIdx);
            m_numVerticesPerMesh.push_back(mesh.vertices().size());

            m_vertices.resizeAndCopy(m_vertices.size() + mesh.vertices().size());
            m_triangles.resizeAndCopy(m_triangles.size() + mesh.triangles().size());

            for (unsigned int i = firstVertexIdx; i < m_vertices.size(); ++i) {
                m_vertices.get()[i] = mesh.vertices()[i - firstVertexIdx];
            }

            for (unsigned int i = firstTriangleIdx; i < m_triangles.size(); ++i) {
                m_triangles.get()[i] = offsetTriangleIndices(mesh.triangles()[i - firstTriangleIdx], firstVertexIdx);
            }

            return addedMeshIdx;
        }

        [[nodiscard]] inline const HostPtr<Vertex3D>& verticesPtr() const
        { return m_vertices; }

        [[nodiscard]] inline const HostPtr<triangleIndices_t>& indicesPtr() const
        { return m_triangles; }

        [[nodiscard]] inline const std::vector<unsigned int>& firstVertexIdxPerMeshVec() const
        { return m_firstVertexIdxPerMesh; }

        [[nodiscard]] inline const std::vector<unsigned int>& numVerticesPerMeshVec() const
        { return m_numVerticesPerMesh; }

    private:
        HostPtr<Vertex3D> m_vertices;
        HostPtr<triangleIndices_t> m_triangles;
        std::vector<unsigned int> m_firstVertexIdxPerMesh;
        std::vector<unsigned int> m_numVerticesPerMesh;

        static triangleIndices_t offsetTriangleIndices(const triangleIndices_t& ti, unsigned int offset)
        {
            return triangleIndices_t(ti.p0 + offset, ti.p1 + offset, ti.p2 + offset);
        }

    }; // Scene

} // Custosh


#endif // CUSTOSH_SCENE_H
