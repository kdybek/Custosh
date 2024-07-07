#ifndef CUSTOSH_SCENE_H
#define CUSTOSH_SCENE_H


#include "utility.h"

namespace Custosh
{
    class Mesh
    {
    public:
        Mesh(std::vector<Vertex3D> vertices, std::vector<triangleIndices_t> triangles)
                : m_vertices(std::move(vertices)), m_triangles(std::move(triangles))
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
        Scene();

        ~Scene();

        // Returns the mesh index withing the scene.
        unsigned int add(const Mesh& mesh);

        void loadVerticesToDev(Vertex3D* devPtr) const;

        void loadTrianglesToDev(triangleIndices_t* devPtr) const;

        [[nodiscard]] unsigned int numVertices() const;

        [[nodiscard]] unsigned int numTriangles() const;

        [[nodiscard]] const std::vector<unsigned int>& firstVertexIdxPerMeshVec() const;

        [[nodiscard]] const std::vector<unsigned int>& numVerticesPerMeshVec() const;

    private:
        class SceneImpl;
        SceneImpl* m_implPtr;

    }; // Scene

} // Custosh


#endif // CUSTOSH_SCENE_H
