#include "scene.h"

#include "internal/gpu_memory.h"

namespace Custosh
{
    class Scene::SceneImpl
    {
    public:
        explicit SceneImpl(const lightSource_t& ls)
                : m_vertices(),
                  m_triangles(),
                  m_lightSource(ls),
                  m_nextMeshIdx(0)
        {
        }

        // Returns the mesh index withing the scene.
        unsigned int add(const Mesh& mesh)
        {
            const unsigned int firstVertexIdx = m_vertices.size();

            for (const auto& vertex: mesh.vertices()) {
                m_vertices.pushBack(meshVertex_t(vertex, m_nextMeshIdx));
            }

            for (const auto& triangle: mesh.triangles()) {
                m_triangles.pushBack(offsetTriangleIndices(triangle, firstVertexIdx));
            }

            return m_nextMeshIdx++;
        }

        void loadVerticesToDev(meshVertex_t* devPtr) const
        {
            m_vertices.hostPtr().loadToDev(devPtr, m_vertices.size());
        }

        void loadTrianglesToDev(triangleIndices_t* devPtr) const
        {
            m_triangles.hostPtr().loadToDev(devPtr, m_triangles.size());
        }

        [[nodiscard]] inline unsigned int numVertices() const
        { return m_vertices.size(); }

        [[nodiscard]] inline unsigned int numTriangles() const
        { return m_triangles.size(); }

        [[nodiscard]] inline unsigned int numMeshes() const
        { return m_nextMeshIdx; }

        [[nodiscard]] inline const lightSource_t& lightSource() const
        { return m_lightSource; }

    private:
        HostDynamicArray<meshVertex_t> m_vertices;
        HostDynamicArray<triangleIndices_t> m_triangles;
        lightSource_t m_lightSource;
        unsigned int m_nextMeshIdx;

        static triangleIndices_t offsetTriangleIndices(const triangleIndices_t& ti, unsigned int offset)
        {
            return triangleIndices_t(ti.p0 + offset, ti.p1 + offset, ti.p2 + offset);
        }

    }; // Scene::SceneImpl

    Scene::Scene(const lightSource_t& ls) : m_implPtr(std::make_unique<SceneImpl>(ls))
    {
    }

    Scene::~Scene() = default;

    unsigned int Scene::add(const Mesh& mesh)
    {
        return m_implPtr->add(mesh);
    }

    void Scene::loadVerticesToDev(meshVertex_t* devPtr) const
    {
        m_implPtr->loadVerticesToDev(devPtr);
    }

    void Scene::loadTrianglesToDev(triangleIndices_t* devPtr) const
    {
        m_implPtr->loadTrianglesToDev(devPtr);
    }

    unsigned int Scene::numVertices() const
    {
        return m_implPtr->numVertices();
    }

    unsigned int Scene::numTriangles() const
    {
        return m_implPtr->numTriangles();
    }

    unsigned int Scene::numMeshes() const
    {
        return m_implPtr->numMeshes();
    }

    const lightSource_t& Scene::lightSource() const
    {
        return m_implPtr->lightSource();
    }

} // Custosh