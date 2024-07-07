#include "scene.h"

#include "gpu_mem_utils.h"

namespace Custosh
{
    class Scene::SceneImpl
    {
    public:
        SceneImpl() : m_vertices(0), m_triangles(0), m_firstVertexIdxPerMesh(), m_numVerticesPerMesh()
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

        void loadVerticesToDev(Vertex3D* devPtr) const
        {
            m_vertices.loadToDev(devPtr);
        }

        void loadTrianglesToDev(triangleIndices_t* devPtr) const
        {
            m_triangles.loadToDev(devPtr);
        }

        [[nodiscard]] unsigned int numVertices() const
        { return m_vertices.size(); }

        [[nodiscard]] unsigned int numTriangles() const
        { return m_triangles.size(); }

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

    }; // Scene::SceneImpl

    Scene::Scene() : m_implPtr(new SceneImpl)
    {
    }

    Scene::~Scene()
    {
        delete m_implPtr;
    }

    unsigned int Scene::add(const Mesh& mesh)
    {
        return m_implPtr->add(mesh);
    }

    void Scene::loadVerticesToDev(Vertex3D* devPtr) const
    {
        m_implPtr->loadVerticesToDev(devPtr);
    }

    void Scene::loadTrianglesToDev(triangleIndices_t* devPtr) const
    {
        m_implPtr->loadTrianglesToDev(devPtr);
    }

    [[nodiscard]] unsigned int Scene::numVertices() const
    {
        return m_implPtr->numVertices();
    }

    [[nodiscard]] unsigned int Scene::numTriangles() const
    {
        return m_implPtr->numTriangles();
    }

    [[nodiscard]] const std::vector<unsigned int>& Scene::firstVertexIdxPerMeshVec() const
    {
        return m_implPtr->firstVertexIdxPerMeshVec();
    }

    [[nodiscard]] const std::vector<unsigned int>& Scene::numVerticesPerMeshVec() const
    {
        return m_implPtr->numVerticesPerMeshVec();
    }

} // Custosh