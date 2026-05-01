////////////////////////////////////////////////////////////////////////////////
// AccuracyBench: compare MaskedOcclusionCulling vs GPU visibility (uint ID buffer).
////////////////////////////////////////////////////////////////////////////////

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cfloat>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <mr-math/math.hpp>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include "cgltf.h"

#include "accuracy_camera.hpp"
#include "dop26.hpp"
#include "usd_load.hpp"
#if ACCBENCH_HAVE_MR_IMPORTER
#include "mr_import_usd.hpp"
#endif

#include "../MaskedOcclusionCulling.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "third_party/stb_image_write.h"

namespace {

using mr::Matr4f;
using mr::Norm3f;
using mr::PackedVec3f;
using mr::Vec3f;
using mr::Vec4f;
using mr::math::Camera;

static bool StrEqIgnoreCaseAscii(const char *a, const char *b) {
	if (!a || !b)
		return a == b;
	for (; *a && *b; ++a, ++b) {
		unsigned char ca = (unsigned char)*a, cb = (unsigned char)*b;
		if (std::tolower(ca) != std::tolower(cb))
			return false;
	}
	return *a == *b;
}

static bool StrPrefEqIgnoreCaseAscii(const char *s, const char *pre) {
	for (; *pre; ++s, ++pre) {
		if (!*s)
			return false;
		if (std::tolower((unsigned char)*s) != std::tolower((unsigned char)*pre))
			return false;
	}
	return true;
}

static bool ParseResolutionSpec(const char *spec, int &outW, int &outH) {
	if (!spec || !*spec)
		return false;
	char *end = nullptr;
	long w = std::strtol(spec, &end, 10);
	if (end == spec || (*end != 'x' && *end != 'X'))
		return false;
	const char *hStart = end + 1;
	if (!*hStart)
		return false;
	long h = std::strtol(hStart, &end, 10);
	if (hStart == end || *end != '\0')
		return false;
	if (w <= 0 || h <= 0 || w > 65536 || h > 65536)
		return false;
	outW = (int)w;
	outH = (int)h;
	return true;
}

static void SnapMocResolution(int &w, int &h) {
	const int w0 = w, h0 = h;
	w = std::max(8, (w / 8) * 8);
	h = std::max(4, (h / 4) * 4);
	if (w != w0 || h != h0)
		std::fprintf(stderr,
		    "Resolution adjusted to %dx%d (MaskedOcclusionCulling: width multiple of 8, height of 4).\n", w, h);
}

const char *kVertSrc = R"(#version 330 core
layout(location = 0) in vec3 aPos;
uniform mat4 uMVP;
void main() {
	gl_Position = uMVP * vec4(aPos, 1.0);
}
)";

/// Reference pass: RGBA32UI id buffer + R32F clip-space w (for MOC ImportPixelDepthBuffer / 1/w).
const char *kVertRefBenchSrc = R"(#version 330 core
layout(location = 0) in vec3 aPos;
uniform mat4 uMVP;
out float vClipW;
void main() {
	vec4 clip = uMVP * vec4(aPos, 1.0);
	gl_Position = clip;
	vClipW = clip.w;
}
)";

const char *kFragRefBenchSrc = R"(#version 330 core
uniform uint uObjectId;
layout(location = 0) out uvec4 mrt;
layout(location = 1) out float oClipW;
in float vClipW;
void main() {
	mrt = uvec4(0u, 0u, 0u, uObjectId);
	oClipW = vClipW;
}
)";

const char *kFragPreviewSrc = R"(#version 330 core
uniform uint uObjectId;
out vec4 FragColor;
vec3 hashRgb(uint id) {
	uvec3 c = uvec3(id * 1597334677u, id * 3812015801u, id * 2764472693u);
	return vec3(c & 255u) / 255.0;
}
void main() {
	vec3 c = hashRgb(uObjectId);
	FragColor = vec4(c * 0.75 + 0.12, 1.0);
}
)";

const char *kLineWorldVertSrc = R"(#version 330 core
layout(location = 0) in vec3 aPos;
uniform mat4 uMVP;
void main() {
	gl_Position = uMVP * vec4(aPos, 1.0);
}
)";

const char *kLineWorldFragSrc = R"(#version 330 core
uniform vec3 uColor;
out vec4 FragColor;
void main() {
	FragColor = vec4(uColor, 1.0);
}
)";

const char *kOverlayNdcVertSrc = R"(#version 330 core
layout(location = 0) in vec2 aNdc;
void main() {
	gl_Position = vec4(aNdc.xy, 0.0, 1.0);
}
)";

const char *kOverlayNdcFragSrc = R"(#version 330 core
uniform vec3 uColor;
uniform float uAlpha;
out vec4 FragColor;
void main() {
	FragColor = vec4(uColor, uAlpha);
}
)";

static void CheckGl(const char *where) {
	GLenum e = glGetError();
	if (e != GL_NO_ERROR)
		fprintf(stderr, "GL error after %s: 0x%x\n", where, (unsigned)e);
}

static GLuint CompileShader(GLenum type, const char *src) {
	GLuint s = glCreateShader(type);
	glShaderSource(s, 1, &src, nullptr);
	glCompileShader(s);
	GLint ok = 0;
	glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
	if (!ok) {
		char buf[2048];
		glGetShaderInfoLog(s, sizeof(buf), nullptr, buf);
		fprintf(stderr, "Shader compile failed:\n%s\n", buf);
		exit(1);
	}
	return s;
}

static GLuint LinkProgram(GLuint vs, GLuint fs) {
	GLuint p = glCreateProgram();
	glAttachShader(p, vs);
	glAttachShader(p, fs);
	glBindAttribLocation(p, 0, "aPos");
	glLinkProgram(p);
	GLint ok = 0;
	glGetProgramiv(p, GL_LINK_STATUS, &ok);
	if (!ok) {
		char buf[2048];
		glGetProgramInfoLog(p, sizeof(buf), nullptr, buf);
		fprintf(stderr, "Program link failed:\n%s\n", buf);
		exit(1);
	}
	glDeleteShader(vs);
	glDeleteShader(fs);
	return p;
}

/// Like `LinkProgram` but attrib location 0 is bound to `attrib0Name` (for overlay line shaders).
static GLuint LinkProgramAttrib0(GLuint vs, GLuint fs, const char *attrib0Name) {
	GLuint p = glCreateProgram();
	glAttachShader(p, vs);
	glAttachShader(p, fs);
	glBindAttribLocation(p, 0, attrib0Name);
	glLinkProgram(p);
	GLint ok = 0;
	glGetProgramiv(p, GL_LINK_STATUS, &ok);
	if (!ok) {
		char buf[2048];
		glGetProgramInfoLog(p, sizeof(buf), nullptr, buf);
		fprintf(stderr, "Program link failed:\n%s\n", buf);
		exit(1);
	}
	glDeleteShader(vs);
	glDeleteShader(fs);
	return p;
}

// mr::Matr4f packs the same logical layout as glm::mat4 in tests (row i of mr = column i of glm).
// Math element M[i][j] sits at mr[j][i]. OpenGL column-major needs out[j*4+i] = M[i][j] = mr[j][i].
static void MrMatrToColumnMajorGl(const Matr4f &m, float out[16]) {
	for (int row = 0; row < 4; ++row)
		for (int col = 0; col < 4; ++col)
			out[col * 4 + row] = m[static_cast<size_t>(col)][static_cast<size_t>(row)];
}

// glTF / column-major 4×4 (cgltf / OpenGL layout) → mr row storage of the same linear map.
static Matr4f Matr4FromColumnMajor(const float *cm) {
	using R4 = mr::Row<float, 4>;
	return Matr4f{
	    R4{cm[0], cm[4], cm[8], cm[12]},
	    R4{cm[1], cm[5], cm[9], cm[13]},
	    R4{cm[2], cm[6], cm[10], cm[14]},
	    R4{cm[3], cm[7], cm[11], cm[15]},
	};
}

struct MeshDopLocal {
	std::vector<accbench::dop26::Dop3f> localVerts;
	std::vector<std::pair<std::uint16_t, std::uint16_t>> edges;
};

struct Mesh {
	std::vector<PackedVec3f> positions;
	std::vector<unsigned int> indices;
	PackedVec3f aabbMin{};
	PackedVec3f aabbMax{};
	/// Built on demand per k (10…30, even) for MOC DOP test modes.
	mutable std::unordered_map<int, MeshDopLocal> dopByK;
};

struct SceneObject {
	std::shared_ptr<Mesh> mesh;
	Matr4f model{Matr4f::identity()};
	uint32_t id = 0;
};

static void ExpandAabb(PackedVec3f &mn, PackedVec3f &mx, const PackedVec3f &p) {
	mn.x = std::min(mn.x, p.x);
	mn.y = std::min(mn.y, p.y);
	mn.z = std::min(mn.z, p.z);
	mx.x = std::max(mx.x, p.x);
	mx.y = std::max(mx.y, p.y);
	mx.z = std::max(mx.z, p.z);
}

static void MeshComputeAabb(Mesh &m) {
	if (m.positions.empty())
		return;
	m.aabbMin = m.positions[0];
	m.aabbMax = m.positions[0];
	for (const auto &p : m.positions)
		ExpandAabb(m.aabbMin, m.aabbMax, p);
}

static void EnsureMeshDopK(const Mesh &m, int k) {
	if (m.positions.empty() || !accbench::dop26::IsSupportedDopK(k))
		return;
	if (m.dopByK.find(k) != m.dopByK.end())
		return;
	MeshDopLocal built;
	accbench::dop26::BuildDopKFromVertexPositions(
	    &m.positions[0].x, m.positions.size(), sizeof(PackedVec3f), k, built.localVerts, built.edges);
	m.dopByK.emplace(k, std::move(built));
}

static Mesh MakeUnitCube() {
	Mesh mesh;
	const float v[8][3] = {
	    {-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
	    {-1, -1, 1}, {1, -1, 1}, {1, 1, 1}, {-1, 1, 1},
	};
	for (auto &c : v)
		mesh.positions.push_back(PackedVec3f{c[0] * 0.5f, c[1] * 0.5f, c[2] * 0.5f});
	const unsigned idx[] = {
	    0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 5, 1, 0, 4, 5,
	    2, 6, 7, 2, 7, 3, 0, 3, 7, 0, 7, 4, 1, 5, 6, 1, 6, 2,
	};
	for (unsigned i : idx)
		mesh.indices.push_back(i);
	MeshComputeAabb(mesh);
	return mesh;
}

static bool LoadObjMeshes(const char *path, std::vector<SceneObject> &out) {
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string warn, err;
	if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, path, nullptr)) {
		fprintf(stderr, "tinyobj LoadObj failed: %s\n", err.c_str());
		return false;
	}
	if (!warn.empty())
		fprintf(stderr, "tinyobj warning: %s\n", warn.c_str());

	out.clear();
	uint32_t nextId = 1;
	for (const auto &shape : shapes) {
		const auto &tm = shape.mesh;
		if (tm.indices.empty())
			continue;

		SceneObject obj;
		obj.id = nextId++;
		obj.model = Matr4f::identity();
		obj.mesh = std::make_shared<Mesh>();

		std::unordered_map<int, unsigned> remap;
		obj.mesh->indices.reserve(tm.indices.size());
		for (size_t i = 0; i < tm.indices.size(); ++i) {
			int vi = tm.indices[i].vertex_index;
			if (vi < 0)
				continue;
			auto it = remap.find(vi);
			if (it == remap.end()) {
				unsigned ni = (unsigned)obj.mesh->positions.size();
				remap[vi] = ni;
				obj.mesh->positions.push_back(PackedVec3f{
				    attrib.vertices[3 * vi + 0],
				    attrib.vertices[3 * vi + 1],
				    attrib.vertices[3 * vi + 2],
				});
			}
			obj.mesh->indices.push_back(remap[vi]);
		}
		MeshComputeAabb(*obj.mesh);
		if (!obj.mesh->indices.empty())
			out.push_back(std::move(obj));
	}
	return !out.empty();
}

static Matr4f CgltfNodeModelMatrix(const cgltf_node *node) {
	std::vector<const cgltf_node *> chain;
	for (const cgltf_node *n = node; n; n = n->parent)
		chain.push_back(n);
	Matr4f world = Matr4f::identity();
	for (int i = (int)chain.size() - 1; i >= 0; --i) {
		cgltf_float lm[16];
		cgltf_node_transform_local(chain[(cgltf_size)i], lm);
		world = world * Matr4FromColumnMajor(lm);
	}
	return world;
}

static bool LoadGltfMeshes(const char *path, std::vector<SceneObject> &out) {
	cgltf_options opts = {};
	cgltf_data *data = nullptr;
	cgltf_result r = cgltf_parse_file(&opts, path, &data);
	if (r != cgltf_result_success) {
		fprintf(stderr, "cgltf_parse_file failed (%d)\n", (int)r);
		return false;
	}
	r = cgltf_load_buffers(&opts, data, path);
	if (r != cgltf_result_success) {
		fprintf(stderr, "cgltf_load_buffers failed (%d)\n", (int)r);
		cgltf_free(data);
		return false;
	}

	out.clear();
	uint32_t nextId = 1;
	// Many glTF scenes reference the same mesh primitive from thousands of nodes; load each
	// primitive once and share geometry to avoid exhausting memory (e.g. large instanced assets).
	std::unordered_map<uint64_t, std::shared_ptr<Mesh>> primMeshes;
	for (cgltf_size ni = 0; ni < data->nodes_count; ++ni) {
		cgltf_node *node = &data->nodes[ni];
		if (!node->mesh)
			continue;
		cgltf_mesh *gm = node->mesh;
		const ptrdiff_t meshIdx = gm - data->meshes;
		if (meshIdx < 0 || meshIdx >= (ptrdiff_t)data->meshes_count)
			continue;
		for (cgltf_size pi = 0; pi < gm->primitives_count; ++pi) {
			cgltf_primitive *prim = &gm->primitives[pi];
			if (prim->type != cgltf_primitive_type_triangles)
				continue;
			const cgltf_accessor *pos =
			    cgltf_find_accessor(prim, cgltf_attribute_type_position, 0);
			if (!pos || pos->count == 0)
				continue;

			const uint64_t primKey =
			    (uint64_t)(uint32_t)meshIdx << 32 | (uint32_t)pi;
			std::shared_ptr<Mesh> geo;
			auto cached = primMeshes.find(primKey);
			if (cached != primMeshes.end()) {
				geo = cached->second;
			} else {
				auto built = std::make_shared<Mesh>();
				built->positions.reserve((size_t)pos->count);
				for (cgltf_size vi = 0; vi < pos->count; ++vi) {
					cgltf_float v[3];
					if (!cgltf_accessor_read_float(pos, vi, v, 3)) {
						fprintf(stderr, "cgltf: position read failed\n");
						cgltf_free(data);
						return false;
					}
					built->positions.push_back(PackedVec3f{v[0], v[1], v[2]});
				}

				if (prim->indices) {
					const cgltf_accessor *ia = prim->indices;
					if (ia->count % 3 != 0)
						continue;
					built->indices.reserve((size_t)ia->count);
					for (cgltf_size ii = 0; ii < ia->count; ++ii) {
						cgltf_size idx = cgltf_accessor_read_index(ia, ii);
						if (idx >= pos->count) {
							fprintf(stderr, "cgltf: index out of range\n");
							cgltf_free(data);
							return false;
						}
						built->indices.push_back((unsigned int)idx);
					}
				} else {
					if (pos->count % 3 != 0)
						continue;
					built->indices.reserve((size_t)pos->count);
					for (cgltf_size ii = 0; ii < pos->count; ++ii)
						built->indices.push_back((unsigned int)ii);
				}

				MeshComputeAabb(*built);
				if (built->indices.empty())
					continue;
				primMeshes.emplace(primKey, built);
				geo = std::move(built);
			}

			SceneObject obj;
			obj.id = nextId++;
			obj.model = CgltfNodeModelMatrix(node);
			obj.mesh = std::move(geo);
			out.push_back(std::move(obj));
		}
	}

	cgltf_free(data);
	if (out.empty())
		fprintf(stderr, "cgltf: no triangle meshes with POSITION found\n");
	return !out.empty();
}

static bool EndsWithIgnoreCase(const char *s, const char *suf) {
	size_t ls = strlen(s), lu = strlen(suf);
	if (ls < lu)
		return false;
	for (size_t i = 0; i < lu; ++i) {
		char a = (char)tolower((unsigned char)s[ls - lu + i]);
		char b = (char)tolower((unsigned char)suf[i]);
		if (a != b)
			return false;
	}
	return true;
}

static void MakeProceduralScene(std::vector<SceneObject> &out) {
	out.clear();
	unsigned n = 24;
	for (unsigned i = 0; i < n; ++i) {
		SceneObject o;
		o.id = i + 1;
		o.mesh = std::make_shared<Mesh>(MakeUnitCube());
		float fx = float((i * 17) % 13) - 6.f;
		float fy = float((i * 7) % 9) - 3.f;
		float fz = -float(i % 10) * 2.2f - 2.f;
		float s = 0.22f + 0.015f * float(i % 5);
		o.model = (Matr4f::identity() * mr::translate(Vec4f{fx, fy, fz, 0.f}))
		    * (Matr4f::identity() * mr::scale(Vec4f{s, s, s, 1.f}));
		out.push_back(std::move(o));
	}
}

static void ObjectWorldCorners(const SceneObject &o, Vec3f wOut[8]) {
	const PackedVec3f &mn = o.mesh->aabbMin;
	const PackedVec3f &mx = o.mesh->aabbMax;
	const Vec3f lc[8] = {
	    {mn.x, mn.y, mn.z}, {mx.x, mn.y, mn.z},
	    {mx.x, mx.y, mn.z}, {mn.x, mx.y, mn.z},
	    {mn.x, mn.y, mx.z}, {mx.x, mn.y, mx.z},
	    {mx.x, mx.y, mx.z}, {mn.x, mx.y, mx.z},
	};
	for (int i = 0; i < 8; ++i) {
		Vec3f p{lc[i].x(), lc[i].y(), lc[i].z()};
		Vec4f h = p * o.model;
		wOut[i] = Vec3f{h.x(), h.y(), h.z()};
	}
}

// Same frustum setup as mr-graphics FPSCamera::frustum_planes() (camera.hpp) + AABB test as in bounds.h
// (transform_bound_box min/max, is_bound_box_not_visible / is_bound_box_frustum_visible).
static void FrustumPlanesMrGraphics(const Matr4f &viewProj, Vec4f planes[6]) {
	Matr4f vp = viewProj.transposed();
	planes[0] = vp[3] + vp[0];
	planes[1] = vp[3] - vp[0];
	planes[2] = vp[3] + vp[1];
	planes[3] = vp[3] - vp[1];
	planes[4] = vp[3] + vp[2];
	planes[5] = vp[3] - vp[2];
	for (int i = 0; i < 6; ++i) {
		Vec3f n{planes[i].x(), planes[i].y(), planes[i].z()};
		float len = n.length();
		if (len > 1e-20f)
			planes[i] = planes[i] * (1.f / len);
	}
}

static void WorldAabbFromCorners(const Vec3f wCorners[8], Vec3f &outMin, Vec3f &outMax) {
	outMin = outMax = wCorners[0];
	for (int i = 1; i < 8; ++i) {
		outMin = Vec3f{std::min(outMin.x(), wCorners[i].x()), std::min(outMin.y(), wCorners[i].y()),
		    std::min(outMin.z(), wCorners[i].z())};
		outMax = Vec3f{std::max(outMax.x(), wCorners[i].x()), std::max(outMax.y(), wCorners[i].y()),
		    std::max(outMax.z(), wCorners[i].z())};
	}
}

// bounds.h: is_bound_box_not_visible — returns true if this plane alone culls the box.
static bool IsBoundBoxNotVisibleMrGraphics(const Vec4f &plane, const Vec3f &bbMin, const Vec3f &bbMax) {
	Vec3f positive{plane.x() >= 0.f ? bbMax.x() : bbMin.x(), plane.y() >= 0.f ? bbMax.y() : bbMin.y(),
	    plane.z() >= 0.f ? bbMax.z() : bbMin.z()};
	Vec3f negative{plane.x() >= 0.f ? bbMin.x() : bbMax.x(), plane.y() >= 0.f ? bbMin.y() : bbMax.y(),
	    plane.z() >= 0.f ? bbMin.z() : bbMax.z()};
	const float dotNeg =
	    plane.x() * negative.x() + plane.y() * negative.y() + plane.z() * negative.z() + plane.w();
	if (dotNeg > 0.f)
		return false;
	const float dotPos =
	    plane.x() * positive.x() + plane.y() * positive.y() + plane.z() * positive.z() + plane.w();
	if (dotPos < 0.f)
		return true;
	return false;
}

static bool IsWorldAabbFrustumVisibleMrGraphics(const Vec4f frustumPlanes[6], const Vec3f wCorners[8]) {
	Vec3f bbMin, bbMax;
	WorldAabbFromCorners(wCorners, bbMin, bbMax);
	for (int i = 0; i < 6; ++i) {
		if (IsBoundBoxNotVisibleMrGraphics(frustumPlanes[i], bbMin, bbMax))
			return false;
	}
	return true;
}

enum class MocOccludeeTestMode {
	/// TestTriangles on full clip-space mesh (accurate, slower).
	Mesh,
	/// TestRect on NDC bounds of the world AABB (fast; more false positives vs GPU).
	AabbScreenRect,
	/// World AABB projected to NDC; convex-hull 2D fan via TestTriangles (fallback TestRect).
	AabbScreenTriangles,
	/// TestRect from k-dop in local space, projected (tighter than AABB rect, cheaper than full mesh).
	DopKScreenRect,
	/// Projected k-dop hull triangulated in screen space and tested via TestTriangles.
	DopKScreenTriangles,
};

static std::string MocTestModeLabel(MocOccludeeTestMode m, int dopK) {
	switch (m) {
	case MocOccludeeTestMode::Mesh:
		return "mesh";
	case MocOccludeeTestMode::AabbScreenRect:
		return "aabb";
	case MocOccludeeTestMode::AabbScreenTriangles:
		return "aabb-tri";
	case MocOccludeeTestMode::DopKScreenRect: {
		char b[32];
		std::snprintf(b, sizeof b, "dop%d", dopK);
		return b;
	}
	case MocOccludeeTestMode::DopKScreenTriangles: {
		char b[32];
		std::snprintf(b, sizeof b, "dop%dtri", dopK);
		return b;
	}
	}
	return "mesh";
}

/// `dop26` / `dop<k>` / `dop<k>tri` with even k in 10…30 (see dop26::IsSupportedDopK).
static bool ParseMocTestString(const char *v, MocOccludeeTestMode &outMode, int &outDopK) {
	if (!v)
		return false;
	if (StrEqIgnoreCaseAscii(v, "mesh")) {
		outMode = MocOccludeeTestMode::Mesh;
		return true;
	}
	if (StrEqIgnoreCaseAscii(v, "aabb")) {
		outMode = MocOccludeeTestMode::AabbScreenRect;
		return true;
	}
	if (StrEqIgnoreCaseAscii(v, "aabb-tri") || StrEqIgnoreCaseAscii(v, "aabb_tri")) {
		outMode = MocOccludeeTestMode::AabbScreenTriangles;
		return true;
	}
	if (StrEqIgnoreCaseAscii(v, "dop26")) {
		outMode = MocOccludeeTestMode::DopKScreenRect;
		outDopK = 26;
		return true;
	}
	if (StrEqIgnoreCaseAscii(v, "dop26tri")) {
		outMode = MocOccludeeTestMode::DopKScreenTriangles;
		outDopK = 26;
		return true;
	}
	if (!StrPrefEqIgnoreCaseAscii(v, "dop"))
		return false;
	const char *r = v + 3;
	std::size_t nlen = std::strlen(r);
	if (nlen == 0)
		return false;
	bool tri = false;
	if (nlen >= 3 && StrEqIgnoreCaseAscii(r + nlen - 3, "tri")) {
		tri = true;
		nlen -= 3;
	}
	if (nlen == 0)
		return false;
	char *endp = nullptr;
	long k = std::strtol(r, &endp, 10);
	if (endp != r + static_cast<std::ptrdiff_t>(nlen))
		return false;
	if (k < 10 || k > 30 || (k & 1) != 0)
		return false;
	if (!accbench::dop26::IsSupportedDopK((int)k))
		return false;
	outDopK = (int)k;
	outMode = tri ? MocOccludeeTestMode::DopKScreenTriangles : MocOccludeeTestMode::DopKScreenRect;
	return true;
}

static Vec4f ClipEdgePointAtW(const Vec4f &a, const Vec4f &b, float wTarget) {
	float wa = a.w(), wb = b.w();
	float t = (wTarget - wa) / (wb - wa);
	return Vec4f{
	    a.x() + t * (b.x() - a.x()),
	    a.y() + t * (b.y() - a.y()),
	    a.z() + t * (b.z() - a.z()),
	    a.w() + t * (wb - wa),
	};
}

/*!
 * Builds a screen-space NDC rect for TestRect from the world AABB: clip-space corners,
 * edge intersections with w = wClip (so boxes crossing the camera plane still get a finite hull).
 * Returns false only if there is no part of the hull with w >= wClip (fully behind); then use
 * VIEW_CULLED, not mesh fallback.
 * outWMinClip: minimum clip w over contributing points (conservative for MOC reversed depth).
 */
static bool TryWorldAabbProjectToTestRect(
    const Vec3f worldCorners[8], const Matr4f &viewProj, float &outXmin, float &outYmin,
    float &outXmax, float &outYmax, float &outWMinClip) {
	// Same scale idea as a small positive clip w — avoids w=0 in NDC; edges are clipped to this plane.
	const float wClip = 1e-4f;
	Vec4f c[8];
	for (int i = 0; i < 8; ++i)
		c[i] = Vec4f{worldCorners[i].x(), worldCorners[i].y(), worldCorners[i].z(), 1.f} * viewProj;

	static const int kEdges[12][2] = {
	    {0, 1}, {1, 2}, {2, 3}, {3, 0},
	    {4, 5}, {5, 6}, {6, 7}, {7, 4},
	    {0, 4}, {1, 5}, {2, 6}, {3, 7},
	};

	float xmin = 0.f, xmax = 0.f, ymin = 0.f, ymax = 0.f;
	float wMin = FLT_MAX;
	bool have = false;

	auto consider = [&](const Vec4f &p) {
		float w = p.w();
		if (w < wClip)
			return;
		float iw = 1.f / w;
		float nx = p.x() * iw;
		float ny = p.y() * iw;
		if (!have) {
			xmin = xmax = nx;
			ymin = ymax = ny;
			wMin = w;
			have = true;
		} else {
			xmin = std::min(xmin, nx);
			xmax = std::max(xmax, nx);
			ymin = std::min(ymin, ny);
			ymax = std::max(ymax, ny);
			wMin = std::min(wMin, w);
		}
	};

	for (int i = 0; i < 8; ++i)
		consider(c[i]);

	for (int e = 0; e < 12; ++e) {
		const Vec4f &a = c[kEdges[e][0]];
		const Vec4f &b = c[kEdges[e][1]];
		float wa = a.w(), wb = b.w();
		float denom = wb - wa;
		if (std::fabs(denom) < 1e-30f)
			continue;
		float t = (wClip - wa) / denom;
		if (t > 0.f && t < 1.f)
			consider(ClipEdgePointAtW(a, b, wClip));
	}

	if (!have)
		return false;
	outXmin = xmin;
	outXmax = xmax;
	outYmin = ymin;
	outYmax = ymax;
	outWMinClip = wMin;
	return true;
}

static bool TryWorldDop26ProjectToTestRect(
    const std::vector<accbench::dop26::Dop3f> &localVerts,
    const std::vector<std::pair<std::uint16_t, std::uint16_t>> &edges,
    const Matr4f &model, const Matr4f &viewProj, float &outXmin, float &outYmin, float &outXmax,
    float &outYmax, float &outWMinClip) {
	if (localVerts.empty())
		return false;
	const float wClip = 1e-4f;
	std::vector<Vec4f> clip(localVerts.size());
	for (std::size_t i = 0; i < localVerts.size(); ++i) {
		Vec3f p{localVerts[i].x, localVerts[i].y, localVerts[i].z};
		Vec4f h = p * model;
		Vec3f wP{h.x(), h.y(), h.z()};
		clip[i] = Vec4f{wP.x(), wP.y(), wP.z(), 1.f} * viewProj;
	}

	float xmin = 0.f, xmax = 0.f, ymin = 0.f, ymax = 0.f, wMin = FLT_MAX;
	bool have = false;
	auto consider = [&](const Vec4f &p) {
		float w = p.w();
		if (w < wClip)
			return;
		float iw = 1.f / w;
		float nx = p.x() * iw;
		float ny = p.y() * iw;
		if (!have) {
			xmin = xmax = nx;
			ymin = ymax = ny;
			wMin = w;
			have = true;
		} else {
			xmin = std::min(xmin, nx);
			xmax = std::max(xmax, nx);
			ymin = std::min(ymin, ny);
			ymax = std::max(ymax, ny);
			wMin = std::min(wMin, w);
		}
	};

	for (const auto &c : clip)
		consider(c);
	for (const auto &e : edges) {
		if (e.first >= clip.size() || e.second >= clip.size())
			continue;
		const Vec4f &a = clip[e.first];
		const Vec4f &b = clip[e.second];
		float wa = a.w(), wb = b.w();
		float denom = wb - wa;
		if (std::fabs(denom) < 1e-30f)
			continue;
		float t = (wClip - wa) / denom;
		if (t > 0.f && t < 1.f)
			consider(ClipEdgePointAtW(a, b, wClip));
	}
	if (!have)
		return false;
	outXmin = xmin;
	outXmax = xmax;
	outYmin = ymin;
	outYmax = ymax;
	outWMinClip = wMin;
	return true;
}

struct NdcPoint2f {
	float x = 0.f;
	float y = 0.f;
};

static bool TryAddUniqueNdcPoint(std::vector<NdcPoint2f> &outPts, float x, float y) {
	const float eps2 = 1e-12f;
	for (const NdcPoint2f &p : outPts) {
		const float dx = p.x - x;
		const float dy = p.y - y;
		if (dx * dx + dy * dy <= eps2)
			return false;
	}
	outPts.push_back(NdcPoint2f{x, y});
	return true;
}

static bool TryWorldAabbProjectToNdcPoints(
    const Vec3f worldCorners[8], const Matr4f &viewProj, std::vector<NdcPoint2f> &outNdc, float &outWMinClip) {
	outNdc.clear();
	outWMinClip = FLT_MAX;
	const float wClip = 1e-4f;
	Vec4f c[8];
	for (int i = 0; i < 8; ++i)
		c[i] = Vec4f{worldCorners[i].x(), worldCorners[i].y(), worldCorners[i].z(), 1.f} * viewProj;
	static const int kEdges[12][2] = {
	    {0, 1}, {1, 2}, {2, 3}, {3, 0},
	    {4, 5}, {5, 6}, {6, 7}, {7, 4},
	    {0, 4}, {1, 5}, {2, 6}, {3, 7},
	};
	auto consider = [&](const Vec4f &p) {
		const float w = p.w();
		if (w < wClip)
			return;
		const float iw = 1.f / w;
		(void)TryAddUniqueNdcPoint(outNdc, p.x() * iw, p.y() * iw);
		outWMinClip = std::min(outWMinClip, w);
	};
	for (int i = 0; i < 8; ++i)
		consider(c[i]);
	for (int e = 0; e < 12; ++e) {
		const Vec4f &a = c[kEdges[e][0]];
		const Vec4f &b = c[kEdges[e][1]];
		const float wa = a.w(), wb = b.w();
		const float denom = wb - wa;
		if (std::fabs(denom) < 1e-30f)
			continue;
		const float t = (wClip - wa) / denom;
		if (t > 0.f && t < 1.f)
			consider(ClipEdgePointAtW(a, b, wClip));
	}
	return !outNdc.empty() && outWMinClip < FLT_MAX;
}

static bool TryWorldDop26ProjectToNdcPoints(
    const std::vector<accbench::dop26::Dop3f> &localVerts,
    const std::vector<std::pair<std::uint16_t, std::uint16_t>> &edges,
    const Matr4f &model, const Matr4f &viewProj, std::vector<NdcPoint2f> &outNdc, float &outWMinClip) {
	outNdc.clear();
	outWMinClip = FLT_MAX;
	if (localVerts.empty())
		return false;
	const float wClip = 1e-4f;
	std::vector<Vec4f> clip(localVerts.size());
	for (std::size_t i = 0; i < localVerts.size(); ++i) {
		Vec3f p{localVerts[i].x, localVerts[i].y, localVerts[i].z};
		Vec4f h = p * model;
		Vec3f wP{h.x(), h.y(), h.z()};
		clip[i] = Vec4f{wP.x(), wP.y(), wP.z(), 1.f} * viewProj;
	}
	auto consider = [&](const Vec4f &p) {
		const float w = p.w();
		if (w < wClip)
			return;
		const float iw = 1.f / w;
		(void)TryAddUniqueNdcPoint(outNdc, p.x() * iw, p.y() * iw);
		outWMinClip = std::min(outWMinClip, w);
	};

	for (const Vec4f &c : clip)
		consider(c);
	for (const auto &e : edges) {
		if (e.first >= clip.size() || e.second >= clip.size())
			continue;
		const Vec4f &a = clip[e.first];
		const Vec4f &b = clip[e.second];
		const float wa = a.w();
		const float wb = b.w();
		const float denom = wb - wa;
		if (std::fabs(denom) < 1e-30f)
			continue;
		const float t = (wClip - wa) / denom;
		if (t > 0.f && t < 1.f)
			consider(ClipEdgePointAtW(a, b, wClip));
	}
	return outNdc.size() >= 1 && outWMinClip < FLT_MAX;
}

static float Cross2D(const NdcPoint2f &o, const NdcPoint2f &a, const NdcPoint2f &b) {
	return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

static bool BuildConvexHull2D(const std::vector<NdcPoint2f> &points, std::vector<NdcPoint2f> &outHull) {
	outHull.clear();
	if (points.size() < 3)
		return false;
	std::vector<NdcPoint2f> sorted = points;
	std::sort(sorted.begin(), sorted.end(), [](const NdcPoint2f &a, const NdcPoint2f &b) {
		return (a.x < b.x) || (a.x == b.x && a.y < b.y);
	});
	std::vector<NdcPoint2f> h;
	h.reserve(sorted.size() * 2);
	for (const NdcPoint2f &p : sorted) {
		while (h.size() >= 2 && Cross2D(h[h.size() - 2], h[h.size() - 1], p) <= 0.f)
			h.pop_back();
		h.push_back(p);
	}
	const std::size_t lowerSize = h.size();
	for (std::size_t i = sorted.size(); i-- > 0;) {
		const NdcPoint2f &p = sorted[i];
		while (h.size() > lowerSize && Cross2D(h[h.size() - 2], h[h.size() - 1], p) <= 0.f)
			h.pop_back();
		h.push_back(p);
	}
	if (h.size() <= 3)
		return false;
	h.pop_back();
	if (h.size() < 3)
		return false;
	outHull.swap(h);
	return true;
}

static bool BuildClipFanFromHull(
    const std::vector<NdcPoint2f> &hull, float wRef, std::vector<float> &outClipVerts,
    std::vector<unsigned> &outTris) {
	outClipVerts.clear();
	outTris.clear();
	if (hull.size() < 3)
		return false;
	const float wSafe = std::max(wRef, 1e-4f);
	outClipVerts.reserve(hull.size() * 4);
	for (const NdcPoint2f &p : hull) {
		outClipVerts.push_back(p.x * wSafe);
		outClipVerts.push_back(p.y * wSafe);
		outClipVerts.push_back(0.f);
		outClipVerts.push_back(wSafe);
	}
	outTris.reserve((hull.size() - 2) * 3);
	for (unsigned i = 1; i + 1 < (unsigned)hull.size(); ++i) {
		outTris.push_back(0u);
		outTris.push_back(i);
		outTris.push_back(i + 1u);
	}
	return outTris.size() >= 3;
}

static float DepthKey(const Vec3f &worldCenter, const Vec3f &camPos) {
	return (worldCenter - camPos).length();
}

static bool MeshIsDrawable(const Mesh &m) {
	return !m.positions.empty() && m.indices.size() >= 3 && (m.indices.size() % 3) == 0;
}

// Default caps at SSE4.1: avoids fragile AVX2 raster paths with some GCC/toolchain combos.
// Override: ACCURACYBENCH_MOC_IMPL=AVX2|AVX512|SSE41|SSE2
static MaskedOcclusionCulling::Implementation MocImplFromEnv() {
	const char *e = std::getenv("ACCURACYBENCH_MOC_IMPL");
	if (!e)
		return MaskedOcclusionCulling::SSE41;
	if (!std::strcmp(e, "AVX512"))
		return MaskedOcclusionCulling::AVX512;
	if (!std::strcmp(e, "AVX2"))
		return MaskedOcclusionCulling::AVX2;
	if (!std::strcmp(e, "SSE41"))
		return MaskedOcclusionCulling::SSE41;
	if (!std::strcmp(e, "SSE2"))
		return MaskedOcclusionCulling::SSE2;
	return MaskedOcclusionCulling::SSE41;
}

static const char *MocResultStr(MaskedOcclusionCulling::CullingResult r) {
	switch (r) {
	case MaskedOcclusionCulling::VISIBLE:
		return "VISIBLE";
	case MaskedOcclusionCulling::OCCLUDED:
		return "OCCLUDED";
	case MaskedOcclusionCulling::VIEW_CULLED:
		return "VIEW_CULLED";
	default:
		return "?";
	}
}

static bool ParseCameraSpec(const char *s, Vec3f &outPos, Vec3f &outDir, Vec3f &outUp) {
	float px, py, pz, dx, dy, dz, ux, uy, uz;
	int n = std::sscanf(s, "%f,%f,%f|%f,%f,%f|%f,%f,%f",
	    &px, &py, &pz, &dx, &dy, &dz, &ux, &uy, &uz);
	if (n != 9)
		n = std::sscanf(s, "((%f,%f,%f),(%f,%f,%f),(%f,%f,%f))",
		    &px, &py, &pz, &dx, &dy, &dz, &ux, &uy, &uz);
	if (n != 9)
		n = std::sscanf(s, "((%f,%f,%f), (%f,%f,%f), (%f,%f,%f))",
		    &px, &py, &pz, &dx, &dy, &dz, &ux, &uy, &uz);
	if (n != 9)
		n = std::sscanf(s,
		    " (( %f , %f , %f ) , ( %f , %f , %f ) , ( %f , %f , %f ) )",
		    &px, &py, &pz, &dx, &dy, &dz, &ux, &uy, &uz);
	if (n != 9) {
		std::fprintf(stderr,
		    "Camera: use --camera='((px,py,pz),(dx,dy,dz),(ux,uy,uz))' (quoted!) or "
		    "--camera=px,py,pz|dx,dy,dz|ux,uy,uz\n"
		    "  position, view direction, up. In bash unquoted ((…)) breaks parsing.\n");
		return false;
	}
	outPos = Vec3f{px, py, pz};
	outDir = Vec3f{dx, dy, dz};
	outUp = Vec3f{ux, uy, uz};
	if (outDir.length() < 1e-9f) {
		std::fprintf(stderr, "Camera: direction vector length is ~0.\n");
		return false;
	}
	if (outUp.length() < 1e-9f) {
		std::fprintf(stderr, "Camera: up vector length is ~0.\n");
		return false;
	}
	// mr-math Camera builds an orthonormal basis from dir×up; near-parallel vectors explode in normalize_unchecked().
	Vec3f cross = outDir.cross(outUp);
	if (cross.length2() < 1e-24f) {
		std::fprintf(stderr, "Camera: direction and up are nearly parallel (|dir×up|² too small).\n");
		return false;
	}
	return true;
}

/// Same 12 edges as `TryWorldAabbProjectToTestRect` corner order from `ObjectWorldCorners`.
static const int kWorldAabbEdges[12][2] = {
    {0, 1}, {1, 2}, {2, 3}, {3, 0},
    {4, 5}, {5, 6}, {6, 7}, {7, 4},
    {0, 4}, {1, 5}, {2, 6}, {3, 7},
};

/*!
 * Fills line geometry for `--visualize-bounds` / `--visualize-bound-projection`.
 * Branches mirror `runBenchmarkPass` occludee test (same fallbacks).
 */
static void FillOccludeeBoundOverlay(
    const SceneObject &o,
    const Mesh &mesh,
    MocOccludeeTestMode mode,
    int mocDopK,
    const Matr4f &vp,
    bool fillWorld,
    bool fillNdc,
    std::vector<Vec3f> &outWorldSegPairs,
    bool &outHasNdc,
    bool &outNdcIsRect,
    float &outRx0,
    float &outRy0,
    float &outRx1,
    float &outRy1,
    std::vector<NdcPoint2f> &outNdcHull) {
	outWorldSegPairs.clear();
	outHasNdc = false;
	outNdcIsRect = false;
	outNdcHull.clear();

	if (fillWorld) {
		if (mode == MocOccludeeTestMode::AabbScreenRect || mode == MocOccludeeTestMode::AabbScreenTriangles) {
			Vec3f wc[8];
			ObjectWorldCorners(o, wc);
			for (int e = 0; e < 12; ++e) {
				outWorldSegPairs.push_back(wc[kWorldAabbEdges[e][0]]);
				outWorldSegPairs.push_back(wc[kWorldAabbEdges[e][1]]);
			}
		} else if (mode != MocOccludeeTestMode::Mesh) {
			EnsureMeshDopK(mesh, mocDopK);
			const MeshDopLocal *dop = nullptr;
			auto dk = mesh.dopByK.find(mocDopK);
			if (dk != mesh.dopByK.end() && !dk->second.localVerts.empty())
				dop = &dk->second;
			if (!dop) {
				Vec3f wc[8];
				ObjectWorldCorners(o, wc);
				for (int e = 0; e < 12; ++e) {
					outWorldSegPairs.push_back(wc[kWorldAabbEdges[e][0]]);
					outWorldSegPairs.push_back(wc[kWorldAabbEdges[e][1]]);
				}
			} else {
				for (const auto &edge : dop->edges) {
					if (edge.first >= dop->localVerts.size() || edge.second >= dop->localVerts.size())
						continue;
					for (int k = 0; k < 2; ++k) {
						const auto &lv = k == 0 ? dop->localVerts[edge.first] : dop->localVerts[edge.second];
						Vec3f pl{lv.x, lv.y, lv.z};
						Vec4f h = pl * o.model;
						outWorldSegPairs.push_back(Vec3f{h.x(), h.y(), h.z()});
					}
				}
			}
		}
	}

	if (!fillNdc)
		return;

	if (mode == MocOccludeeTestMode::AabbScreenRect) {
		Vec3f wc[8];
		ObjectWorldCorners(o, wc);
		float rwMin = 0.f;
		if (TryWorldAabbProjectToTestRect(wc, vp, outRx0, outRy0, outRx1, outRy1, rwMin)) {
			(void)rwMin;
			outHasNdc = true;
			outNdcIsRect = true;
		}
	} else if (mode == MocOccludeeTestMode::AabbScreenTriangles) {
		Vec3f wc[8];
		ObjectWorldCorners(o, wc);
		std::vector<NdcPoint2f> ndcPoints;
		float wRef = FLT_MAX;
		bool usedHull = false;
		if (TryWorldAabbProjectToNdcPoints(wc, vp, ndcPoints, wRef)) {
			std::vector<NdcPoint2f> hull;
			if (BuildConvexHull2D(ndcPoints, hull)) {
				outNdcHull = std::move(hull);
				outHasNdc = true;
				outNdcIsRect = false;
				usedHull = true;
			}
		}
		if (!usedHull) {
			float rwMin = 0.f;
			if (TryWorldAabbProjectToTestRect(wc, vp, outRx0, outRy0, outRx1, outRy1, rwMin)) {
				(void)rwMin;
				outHasNdc = true;
				outNdcIsRect = true;
			}
		}
	} else if (mode == MocOccludeeTestMode::DopKScreenRect) {
		EnsureMeshDopK(mesh, mocDopK);
		const MeshDopLocal *dop = nullptr;
		auto dk = mesh.dopByK.find(mocDopK);
		if (dk != mesh.dopByK.end() && !dk->second.localVerts.empty())
			dop = &dk->second;
		float rwMin = 0.f;
		bool got = false;
		if (!dop) {
			Vec3f wc[8];
			ObjectWorldCorners(o, wc);
			got = TryWorldAabbProjectToTestRect(wc, vp, outRx0, outRy0, outRx1, outRy1, rwMin);
		} else {
			got = TryWorldDop26ProjectToTestRect(
			    dop->localVerts, dop->edges, o.model, vp, outRx0, outRy0, outRx1, outRy1, rwMin);
			if (!got) {
				Vec3f wc[8];
				ObjectWorldCorners(o, wc);
				got = TryWorldAabbProjectToTestRect(wc, vp, outRx0, outRy0, outRx1, outRy1, rwMin);
			}
		}
		if (got) {
			(void)rwMin;
			outHasNdc = true;
			outNdcIsRect = true;
		}
	} else if (mode == MocOccludeeTestMode::DopKScreenTriangles) {
		EnsureMeshDopK(mesh, mocDopK);
		const MeshDopLocal *dop = nullptr;
		auto dk = mesh.dopByK.find(mocDopK);
		if (dk != mesh.dopByK.end() && !dk->second.localVerts.empty())
			dop = &dk->second;
		bool usedHull = false;
		if (dop) {
			std::vector<NdcPoint2f> ndcPoints;
			float wRef = FLT_MAX;
			if (TryWorldDop26ProjectToNdcPoints(
			        dop->localVerts, dop->edges, o.model, vp, ndcPoints, wRef)) {
				std::vector<NdcPoint2f> hull;
				if (BuildConvexHull2D(ndcPoints, hull)) {
					outNdcHull = std::move(hull);
					outHasNdc = true;
					outNdcIsRect = false;
					usedHull = true;
				}
			}
		}
		if (!usedHull) {
			float rwMin = 0.f;
			bool got = false;
			if (!dop) {
				Vec3f wc[8];
				ObjectWorldCorners(o, wc);
				got = TryWorldAabbProjectToTestRect(wc, vp, outRx0, outRy0, outRx1, outRy1, rwMin);
			} else {
				got = TryWorldDop26ProjectToTestRect(
				    dop->localVerts, dop->edges, o.model, vp, outRx0, outRy0, outRx1, outRy1, rwMin);
				if (!got) {
					Vec3f wc[8];
					ObjectWorldCorners(o, wc);
					got = TryWorldAabbProjectToTestRect(wc, vp, outRx0, outRy0, outRx1, outRy1, rwMin);
				}
			}
			if (got) {
				(void)rwMin;
				outHasNdc = true;
				outNdcIsRect = true;
			}
		}
	}
}

} // namespace

static double g_scrollAccum = 0.0;

static void ScrollCallback(GLFWwindow *, double, double yoff) {
	g_scrollAccum += yoff;
}

static bool FillObjectsFromUsd(std::vector<accbench::usd::MeshBuffers> &umb,
    std::vector<accbench::usd::Instance> &uinst, std::vector<SceneObject> &objects) {
	objects.clear();
	uint32_t nextId = 1;
	std::vector<std::shared_ptr<Mesh>> meshByIdx(umb.size());
	for (size_t mi = 0; mi < umb.size(); ++mi) {
		auto m = std::make_shared<Mesh>();
		m->positions.reserve(umb[mi].positions.size());
		for (const auto &p : umb[mi].positions)
			m->positions.push_back(PackedVec3f{p[0], p[1], p[2]});
		m->indices = std::move(umb[mi].indices);
		MeshComputeAabb(*m);
		meshByIdx[mi] = std::move(m);
	}
	for (const auto &inst : uinst) {
		if (inst.meshIndex < 0 || static_cast<size_t>(inst.meshIndex) >= meshByIdx.size())
			continue;
		SceneObject obj;
		obj.id = nextId++;
		obj.model = Matr4FromColumnMajor(inst.modelColumnMajor);
		obj.mesh = meshByIdx[static_cast<size_t>(inst.meshIndex)];
		if (!obj.mesh->indices.empty())
			objects.push_back(std::move(obj));
	}
	return !objects.empty();
}

#if ACCBENCH_HAVE_MR_IMPORTER
static bool TryLoadPathViaMrImporter(const char *path, std::vector<SceneObject> &objects, std::string &err) {
	err.clear();
	std::vector<accbench::usd::MeshBuffers> umb;
	std::vector<accbench::usd::Instance> uinst;
	if (!accbench::mrimp::LoadStage(path, umb, uinst, err))
		return false;
	if (!FillObjectsFromUsd(umb, uinst, objects)) {
		err = "mr-importer: no drawable objects after import";
		return false;
	}
	return true;
}
#endif

static bool ParsePositiveFloatArg(const char *s, float &out, const char *flagName) {
	char *end = nullptr;
	out = std::strtof(s, &end);
	if (end == s || *end != '\0') {
		std::fprintf(stderr, "%s requires a single floating-point value.\n", flagName);
		return false;
	}
	if (!(out > 0.f) || out >= 1e9f) {
		std::fprintf(stderr, "%s must be a positive finite value (suggested range 1e-6 … 1e8).\n", flagName);
		return false;
	}
	return true;
}

static bool ParseMaxFramesArg(const char *s, int &out, const char *flagName) {
	char *end = nullptr;
	unsigned long v = std::strtoul(s, &end, 10);
	if (end == s || *end != '\0') {
		std::fprintf(stderr, "%s requires a positive integer.\n", flagName);
		return false;
	}
	if (v < 1ul || v > 1000000ul) {
		std::fprintf(stderr, "%s must be in range 1..1000000.\n", flagName);
		return false;
	}
	out = (int)v;
	return true;
}

/// `--save-depth-frame=N`: 0-based benchmark pass index; writes `depthN.png` (per-pixel z from MOC hi-z).
static bool ParseSaveDepthFrameArg(const char *s, int &out, const char *flagName) {
	char *end = nullptr;
	unsigned long v = std::strtoul(s, &end, 10);
	if (end == s || *end != '\0') {
		std::fprintf(stderr, "%s requires a non-negative integer (0 = first pass).\n", flagName);
		return false;
	}
	if (v > 10000000ul) {
		std::fprintf(stderr, "%s must be <= 10000000.\n", flagName);
		return false;
	}
	out = (int)v;
	return true;
}

/// Same tonemap idea as Example/ExampleMain (Intel sample): linear map w in [min,max] → gray 32…223.
static void TonemapMocDepthToRgb(const float *depth, unsigned char *imageRgb, int w, int h) {
	float minW = FLT_MAX, maxW = 0.f;
	for (int i = 0; i < w * h; ++i) {
		if (depth[i] > 0.f) {
			minW = std::min(minW, depth[i]);
			maxW = std::max(maxW, depth[i]);
		}
	}
	const float range = maxW - minW;
	for (int i = 0; i < w * h; ++i) {
		int intensity = 0;
		if (depth[i] > 0.f && range > 1e-20f) {
			const float t = (depth[i] - minW) / range;
			intensity = (int)(223.0 * (double)t + 32.0);
			if (intensity < 0)
				intensity = 0;
			else if (intensity > 255)
				intensity = 255;
		}
		imageRgb[i * 3 + 0] = (unsigned char)intensity;
		imageRgb[i * 3 + 1] = (unsigned char)intensity;
		imageRgb[i * 3 + 2] = (unsigned char)intensity;
	}
}

static constexpr char kAccbenchSaveDepthPrefix[] = "--save-depth-frame=";

static bool SaveMocHizDepthPng(MaskedOcclusionCulling *moc, int w, int h, const char *pathOut) {
	std::vector<float> depth((size_t)w * (size_t)h);
	// USE_D3D=0: flipY=true so first scanline = top of screen like GL viewport / PNG viewers (see
	// MaskedOcclusionCullingCommon.inl; FrameRecorderPlayer uses true).
	moc->ComputePixelDepthBuffer(depth.data(), true);
	std::vector<unsigned char> rgb((size_t)w * (size_t)h * 3);
	TonemapMocDepthToRgb(depth.data(), rgb.data(), w, h);
	if (!stbi_write_png(pathOut, w, h, 3, rgb.data(), w * 3)) {
		std::fprintf(stderr, "stbi_write_png failed: %s\n", pathOut);
		return false;
	}
	return true;
}

/// Read GL_COLOR_ATTACHMENT1 (R32F clip w); convert rows GL bottom-first → top-first rcpW for MOC ImportPixelDepthBuffer(..., true).
static void AccBenchReadClipWToTopFirstRcpW(int w, int h, std::vector<float> &outTopFirstRcpW) {
	std::vector<float> tmp((size_t)w * (size_t)h);
	glReadBuffer(GL_COLOR_ATTACHMENT1);
	glReadPixels(0, 0, w, h, GL_RED, GL_FLOAT, tmp.data());
	outTopFirstRcpW.resize((size_t)w * (size_t)h);
	for (int y = 0; y < h; y++) {
		const float *srcRow = tmp.data() + (size_t)(h - 1 - y) * (size_t)w;
		float *dstRow = outTopFirstRcpW.data() + (size_t)y * (size_t)w;
		for (int x = 0; x < w; x++) {
			const float cw = srcRow[x];
			dstRow[x] = (std::isfinite(cw) && cw > 1e-8f) ? (1.f / cw) : 0.f;
		}
	}
}

int main(int argc, char **argv) {
	bool headless = false;
	bool perObjectReport = false;
	bool usePrevFramePrev = false;
	/// When false (default): GPU id readback draws only MOC VISIBLE (+ frustum), matching typical submit.
	/// `--gpu-draw-always`: draw all frustum for reference buffer / FN pixels (ignores MOC query outcome).
	bool gpuDrawAlways = false;
	const char *objPath = nullptr;
	const char *cameraSpec = nullptr;
	int fbW = 1280;
	int fbH = 720;
	float clipNear = 0.01f;
	float clipFar = 1000.f;
	/// 0: default (unbounded preview; headless: single benchmark pass).  N>0: see --max-frames=.
	int maxFramesLimit = 0;
	/// -1: off. Else save `depthN.png` on pass index N (0-based) after MOC occluder rasterization.
	int saveMocDepthFrame = -1;
	bool visualizeBounds = false;
	bool visualizeBoundProjection = false;
	MocOccludeeTestMode mocOccludeeTest = MocOccludeeTestMode::Mesh;
	int mocDopK = 26;
	const char *mocTestEnv = std::getenv("ACCURACYBENCH_MOC_TEST");
	if (mocTestEnv && mocTestEnv[0]) {
		if (!ParseMocTestString(mocTestEnv, mocOccludeeTest, mocDopK)) {
			std::fprintf(stderr,
			    "ACCURACYBENCH_MOC_TEST: expected 'mesh', 'aabb', 'aabb-tri', 'dop<k>', or 'dop<k>tri' "
			    "with even k in 10…30 (got '%s'); using mesh.\n",
			    mocTestEnv);
			mocOccludeeTest = MocOccludeeTestMode::Mesh;
		}
	}
	for (int i = 1; i < argc; ++i) {
		if (!std::strncmp(argv[i], "--moc-test=", 11)) {
			const char *v = argv[i] + 11;
			if (!ParseMocTestString(v, mocOccludeeTest, mocDopK)) {
				std::fprintf(stderr,
				    "--moc-test= expects 'mesh', 'aabb', 'aabb-tri', 'dop<k>', or 'dop<k>tri' with even k in "
				    "10…30 (TestTriangles full mesh; TestRect from AABB; 2D hull from projected AABB; "
				    "TestRect from k-dop in NDC; 2D hull fan from projected k-dop). Legacy aliases: dop26, dop26tri.\n");
				return 1;
			}
			continue;
		}
		if (!std::strcmp(argv[i], "--headless")) {
			headless = true;
			continue;
		}
		if (!std::strcmp(argv[i], "--per-object")) {
			perObjectReport = true;
			continue;
		}
		if (!std::strncmp(argv[i], "--resolution=", 13)) {
			if (!ParseResolutionSpec(argv[i] + 13, fbW, fbH)) {
				std::fprintf(stderr, "Invalid --resolution=WxH (example: --resolution=1920x1080).\n");
				return 1;
			}
			SnapMocResolution(fbW, fbH);
			continue;
		}
		if (!std::strncmp(argv[i], "--resoltuion=", 13)) {
			if (!ParseResolutionSpec(argv[i] + 13, fbW, fbH)) {
				std::fprintf(stderr, "Invalid --resoltuion=WxH (typo alias for --resolution=).\n");
				return 1;
			}
			SnapMocResolution(fbW, fbH);
			continue;
		}
		if (!std::strcmp(argv[i], "--resolution")) {
			if (i + 1 >= argc) {
				std::fprintf(stderr, "--resolution requires WxH.\n");
				return 1;
			}
			if (!ParseResolutionSpec(argv[++i], fbW, fbH)) {
				std::fprintf(stderr, "Invalid --resolution WxH (example: 1920x1080).\n");
				return 1;
			}
			SnapMocResolution(fbW, fbH);
			continue;
		}
		if (!std::strncmp(argv[i], "--camera=", 9)) {
			cameraSpec = argv[i] + 9;
			continue;
		}
		if (!std::strcmp(argv[i], "--camera")) {
			if (i + 1 >= argc) {
				std::fprintf(stderr, "--camera requires a value.\n");
				return 1;
			}
			cameraSpec = argv[++i];
			continue;
		}
		if (!std::strncmp(argv[i], "--near=", 7)) {
			if (!ParsePositiveFloatArg(argv[i] + 7, clipNear, "--near="))
				return 1;
			continue;
		}
		if (!std::strcmp(argv[i], "--near")) {
			if (i + 1 >= argc) {
				std::fprintf(stderr, "--near requires a value.\n");
				return 1;
			}
			if (!ParsePositiveFloatArg(argv[++i], clipNear, "--near"))
				return 1;
			continue;
		}
		if (!std::strncmp(argv[i], "--far=", 6)) {
			if (!ParsePositiveFloatArg(argv[i] + 6, clipFar, "--far="))
				return 1;
			continue;
		}
		if (!std::strcmp(argv[i], "--far")) {
			if (i + 1 >= argc) {
				std::fprintf(stderr, "--far requires a value.\n");
				return 1;
			}
			if (!ParsePositiveFloatArg(argv[++i], clipFar, "--far"))
				return 1;
			continue;
		}
		// "--max-frames=" is 13 bytes; n must be 13 or strncmp also compares first digit to '\0' → no match.
		if (!std::strncmp(argv[i], "--max-frames=", 13)) {
			if (!ParseMaxFramesArg(argv[i] + 13, maxFramesLimit, "--max-frames="))
				return 1;
			continue;
		}
		if (!std::strcmp(argv[i], "--max-frames")) {
			if (i + 1 >= argc) {
				std::fprintf(stderr, "--max-frames requires a positive integer.\n");
				return 1;
			}
			if (!ParseMaxFramesArg(argv[++i], maxFramesLimit, "--max-frames"))
				return 1;
			continue;
		}
		if (!std::strncmp(argv[i], kAccbenchSaveDepthPrefix, sizeof(kAccbenchSaveDepthPrefix) - 1)) {
			if (!ParseSaveDepthFrameArg(argv[i] + sizeof(kAccbenchSaveDepthPrefix) - 1, saveMocDepthFrame,
				"--save-depth-frame="))
				return 1;
			continue;
		}
		if (!std::strcmp(argv[i], "--save-depth-frame")) {
			if (i + 1 >= argc) {
				std::fprintf(stderr, "--save-depth-frame requires a non-negative integer.\n");
				return 1;
			}
			if (!ParseSaveDepthFrameArg(argv[++i], saveMocDepthFrame, "--save-depth-frame"))
				return 1;
			continue;
		}
		if (!std::strcmp(argv[i], "--visualize-bounds")) {
			visualizeBounds = true;
			continue;
		}
		if (!std::strcmp(argv[i], "--visualize-bound-projection") ||
		    !std::strcmp(argv[i], "--visualize-bounds-projection")) {
			visualizeBoundProjection = true;
			continue;
		}
		if (!std::strcmp(argv[i], "--use-prev-frame-prev")) {
			usePrevFramePrev = true;
			continue;
		}
		if (!std::strcmp(argv[i], "--gpu-draw-always")) {
			gpuDrawAlways = true;
			continue;
		}
		if (argv[i][0] != '-' && !objPath)
			objPath = argv[i];
	}

	if (headless && (visualizeBounds || visualizeBoundProjection)) {
		std::fprintf(stderr,
		    "ACCURACYBENCH: --visualize-bounds / --visualize-bound-projection require the preview window; "
		    "ignored with --headless.\n");
	}
	if (usePrevFramePrev) {
		std::fprintf(stderr,
		    "ACCURACYBENCH: --use-prev-frame-prev → GPU draw all frustum first → ImportPixelDepthBuffer from clip.w "
		    "(~1/w).\n");
		if (!gpuDrawAlways)
			std::fprintf(stderr,
			    "  Default id readback: after MOC queries, second draw clears FBO and submits only MOC VISIBLE "
			    "(culled). FN pixel metric unavailable.\n");
		else
			std::fprintf(stderr,
			    "  --gpu-draw-always: same pass feeds Hi-Z Import + id readback (full frustum overdraw reference).\n");
	}
	if (!gpuDrawAlways && !usePrevFramePrev) {
		std::fprintf(stderr,
		    "ACCURACYBENCH: default GPU reference draw = culled visible-only after MOC queries (matches typical "
		    "submit). Use --gpu-draw-always for full frustum readback / FN pixels.\n");
	}
	if (gpuDrawAlways && !usePrevFramePrev) {
		std::fprintf(stderr,
		    "ACCURACYBENCH: --gpu-draw-always → reference draw includes all frustum instances regardless of MOC.\n");
	}

	std::vector<SceneObject> objects;
	if (objPath) {
		bool ok = false;
		if (EndsWithIgnoreCase(objPath, ".glb") || EndsWithIgnoreCase(objPath, ".gltf")) {
			ok = false;
#if ACCBENCH_HAVE_MR_IMPORTER
			{
				std::string ue;
				if (TryLoadPathViaMrImporter(objPath, objects, ue))
					ok = true;
				else if (!ue.empty())
					fprintf(stderr, "glTF (mr-importer): %s — trying cgltf…\n", ue.c_str());
			}
#endif
			if (!ok)
				ok = LoadGltfMeshes(objPath, objects);
		} else if (EndsWithIgnoreCase(objPath, ".usd") || EndsWithIgnoreCase(objPath, ".usda") ||
		    EndsWithIgnoreCase(objPath, ".usdc") || EndsWithIgnoreCase(objPath, ".usdz")) {
			std::vector<accbench::usd::MeshBuffers> umb;
			std::vector<accbench::usd::Instance> uinst;
			std::string uw, ue;
			ok = false;
#if ACCBENCH_HAVE_MR_IMPORTER
			if (TryLoadPathViaMrImporter(objPath, objects, ue))
				ok = true;
			if (!ok) {
				if (!ue.empty())
					fprintf(stderr, "USD (mr-importer): %s — trying TinyUSDZ path…\n", ue.c_str());
				umb.clear();
				uinst.clear();
				ue.clear();
			}
#endif
			if (!ok && accbench::usd::LoadStage(objPath, umb, uinst, uw, ue)) {
				if (!uw.empty())
					fprintf(stderr, "USD warn: %s\n", uw.c_str());
				ok = FillObjectsFromUsd(umb, uinst, objects);
			} else if (!ok) {
				fprintf(stderr, "USD load failed: %s\n", ue.c_str());
			}
		} else {
			ok = LoadObjMeshes(objPath, objects);
		}
		if (!ok) {
			fprintf(stderr, "Falling back to procedural scene.\n");
			MakeProceduralScene(objects);
		}
	} else {
		MakeProceduralScene(objects);
	}

	if (objects.empty()) {
		fprintf(stderr, "No objects in scene.\n");
		return 1;
	}

	if (!(clipFar > clipNear)) {
		std::fprintf(stderr, "Clip planes: need --far > --near (got near=%g far=%g).\n",
		    (double)clipNear, (double)clipFar);
		return 1;
	}

	accbench::FpsCamera fps;
	const float aspect0 = float(fbW) / float(fbH);
	fps.configureProjection(aspect0, clipNear, clipFar);
	if (cameraSpec) {
		std::fprintf(stderr, "Parsing --camera: %s\n", cameraSpec);
		Vec3f cpos, cdir, cup;
		if (!ParseCameraSpec(cameraSpec, cpos, cdir, cup))
			return 1;
		fps.cam() = mr::math::Camera<float>(cpos, cdir.normalized_unchecked(), cup.normalized_unchecked());
		fps.configureProjection(aspect0, clipNear, clipFar);
	}

	const float nearP = fps.cam().projection().distance;

	if (!glfwInit()) {
		fprintf(stderr, "glfwInit failed\n");
		return 1;
	}
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_VISIBLE, headless ? GLFW_FALSE : GLFW_TRUE);
	GLFWwindow *win =
	    glfwCreateWindow(fbW, fbH, "AccuracyBench - preview (close to exit)", nullptr, nullptr);
	if (!win) {
		fprintf(stderr, "glfwCreateWindow failed\n");
		return 1;
	}
	glfwMakeContextCurrent(win);
	glfwSetScrollCallback(win, ScrollCallback);
	// if (!headless)
	// 	glfwSetInputMode(win, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "glewInit failed\n");
		return 1;
	}
	glGetError();

	GLuint vsRef = CompileShader(GL_VERTEX_SHADER, kVertRefBenchSrc);
	GLuint fsRef = CompileShader(GL_FRAGMENT_SHADER, kFragRefBenchSrc);
	GLuint prog = LinkProgram(vsRef, fsRef);
	GLint locMvp = glGetUniformLocation(prog, "uMVP");
	GLint locId = glGetUniformLocation(prog, "uObjectId");
	if (locMvp < 0 || locId < 0) {
		fprintf(stderr, "Missing uniforms\n");
		return 1;
	}

	GLuint vsPrev = CompileShader(GL_VERTEX_SHADER, kVertSrc);
	GLuint fsPrev = CompileShader(GL_FRAGMENT_SHADER, kFragPreviewSrc);
	GLuint progPreview = LinkProgram(vsPrev, fsPrev);
	GLint locMvpPrev = glGetUniformLocation(progPreview, "uMVP");
	GLint locIdPrev = glGetUniformLocation(progPreview, "uObjectId");
	if (locMvpPrev < 0 || locIdPrev < 0) {
		fprintf(stderr, "Missing preview uniforms\n");
		return 1;
	}

	GLuint vsLw = CompileShader(GL_VERTEX_SHADER, kLineWorldVertSrc);
	GLuint fsLw = CompileShader(GL_FRAGMENT_SHADER, kLineWorldFragSrc);
	GLuint progLineWorld = LinkProgram(vsLw, fsLw);
	GLint locLineMvp = glGetUniformLocation(progLineWorld, "uMVP");
	GLint locLineColor = glGetUniformLocation(progLineWorld, "uColor");
	if (locLineMvp < 0 || locLineColor < 0) {
		fprintf(stderr, "Missing bound-line shader uniforms.\n");
		return 1;
	}
	GLuint vsOv = CompileShader(GL_VERTEX_SHADER, kOverlayNdcVertSrc);
	GLuint fsOv = CompileShader(GL_FRAGMENT_SHADER, kOverlayNdcFragSrc);
	GLuint progOverlayNdc = LinkProgramAttrib0(vsOv, fsOv, "aNdc");
	GLint locNdcColor = glGetUniformLocation(progOverlayNdc, "uColor");
	GLint locNdcAlpha = glGetUniformLocation(progOverlayNdc, "uAlpha");
	if (locNdcColor < 0 || locNdcAlpha < 0) {
		fprintf(stderr, "Missing NDC overlay uniforms (need uColor + uAlpha).\n");
		return 1;
	}

	GLuint vao = 0, vbo = 0, ebo = 0, lineVbo = 0;
	glGenVertexArrays(1, &vao);
	glGenBuffers(1, &vbo);
	glGenBuffers(1, &ebo);
	glGenBuffers(1, &lineVbo);

	GLuint fbo = 0, colorTex = 0, clipWTex = 0, depthRb = 0;
	glGenFramebuffers(1, &fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	glGenTextures(1, &colorTex);
	glBindTexture(GL_TEXTURE_2D, colorTex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32UI, fbW, fbH, 0, GL_RGBA_INTEGER, GL_UNSIGNED_INT, nullptr);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorTex, 0);
	glGenTextures(1, &clipWTex);
	glBindTexture(GL_TEXTURE_2D, clipWTex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, fbW, fbH, 0, GL_RED, GL_FLOAT, nullptr);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, clipWTex, 0);
	const GLenum kDrawBufs[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
	glDrawBuffers(2, kDrawBufs);
	glGenRenderbuffers(1, &depthRb);
	glBindRenderbuffer(GL_RENDERBUFFER, depthRb);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, fbW, fbH);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthRb);
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
		fprintf(stderr, "FBO incomplete\n");
		return 1;
	}
	CheckGl("FBO setup");

	enum class BenchPrintStyle { Full, LiveFour };

	struct AccBenchPassResult {
		unsigned nAll = 0;
		unsigned nFrustum = 0;
		unsigned nMocVisible = 0;
		unsigned nGpuVisible = 0;
		unsigned fp = 0;
		unsigned fn = 0;
		/// Frustum subset where MOC VISIBLE and GPU had ≥1 id pixel (cross-check: agree+fp=mocVis, agree+fn=gpuIds).
		unsigned agreeVisFrustum = 0;
		double mocBufferMs = 0;
		double mocQueryMs = 0;
		/// Offscreen RGBA32UI + depth: clear, upload, draw, glFinish (no readback).
		double gpuRefMs = 0;
		/// glReadPixels RGBA32UI readback (benchmark ground-truth fetch).
		double readPixelsMs = 0;
		/// Wall time: Create → after FP/FN stats (excludes printf / Destroy).
		double passWallMs = 0;
	};

	std::vector<unsigned char> guiBenchMocVisible(objects.size(), 0);
	std::vector<unsigned char> guiPrevMocVisibleForOverlay(objects.size(), 0);

	std::vector<float> gpuClipScratchRcpW;

	auto drawGpuReferenceFill = [&](const Matr4f &vp, const std::vector<char> *visibleMask) {
		glViewport(0, 0, fbW, fbH);
		glBindFramebuffer(GL_FRAMEBUFFER, fbo);
		const GLuint clearZ[4] = {0, 0, 0, 0};
		glClearBufferuiv(GL_COLOR, 0, clearZ);
		const GLfloat clearClipW[1] = {0.f};
		glClearBufferfv(GL_COLOR, 1, clearClipW);
		glClearDepth(1.0);
		glClear(GL_DEPTH_BUFFER_BIT);
		glEnable(GL_DEPTH_TEST);
		glDepthFunc(GL_LESS);
		glUseProgram(prog);
		glBindVertexArray(vao);
		float mvpColLocal[16];
		Vec4f frustumPlanesGpu[6];
		FrustumPlanesMrGraphics(vp, frustumPlanesGpu);
		for (size_t i = 0; i < objects.size(); ++i) {
			const SceneObject &o = objects[i];
			if (!MeshIsDrawable(*o.mesh))
				continue;
			Vec3f wcornersGpu[8];
			ObjectWorldCorners(o, wcornersGpu);
			if (!IsWorldAabbFrustumVisibleMrGraphics(frustumPlanesGpu, wcornersGpu))
				continue;
			if (visibleMask) {
				if (i >= visibleMask->size() || !(*visibleMask)[i])
					continue;
			}
			Matr4f mvp = o.model * vp;
			MrMatrToColumnMajorGl(mvp, mvpColLocal);
			glUniformMatrix4fv(locMvp, 1, GL_FALSE, mvpColLocal);
			glUniform1ui(locId, o.id);
			glBindBuffer(GL_ARRAY_BUFFER, vbo);
			glBufferData(GL_ARRAY_BUFFER, o.mesh->positions.size() * sizeof(PackedVec3f),
			    o.mesh->positions.data(), GL_STREAM_DRAW);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, o.mesh->indices.size() * sizeof(unsigned),
			    o.mesh->indices.data(), GL_STREAM_DRAW);
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(PackedVec3f), (void *)0);
			glDrawElements(GL_TRIANGLES, (GLsizei)o.mesh->indices.size(), GL_UNSIGNED_INT, nullptr);
		}
		CheckGl("drawGpuReferenceFill");
	};

	auto runBenchmarkPass = [&](int passFrameIndex, const Matr4f &vp, const Vec3f &camPosForSort,
	                    BenchPrintStyle printStyle) -> AccBenchPassResult {
		const auto tPassWall0 = std::chrono::steady_clock::now();
		MaskedOcclusionCulling *moc = usePrevFramePrev ? MaskedOcclusionCulling::CreateFromDepth(MocImplFromEnv())
		                                                : MaskedOcclusionCulling::Create(MocImplFromEnv());
		moc->SetResolution((unsigned)fbW, (unsigned)fbH);
		moc->SetNearClipPlane(fps.cam().projection().distance);

		double gpuRefMs = 0.0;
		double mocBufferMs = 0.0;

		float mvpCol[16];

		if (usePrevFramePrev) {
			/// One GPU draw for this vp: same color/depth/clip.w buffer drives Import + later readPixels.
			const auto tGpuRef0 = std::chrono::steady_clock::now();
			drawGpuReferenceFill(vp, nullptr);
			glFinish();
			const auto tGpuRef1 = std::chrono::steady_clock::now();
			gpuRefMs = std::chrono::duration<double, std::milli>(tGpuRef1 - tGpuRef0).count();

			const auto tMocBuffer0 = std::chrono::steady_clock::now();
			AccBenchReadClipWToTopFirstRcpW(fbW, fbH, gpuClipScratchRcpW);
			moc->ClearBuffer();
			moc->ImportPixelDepthBuffer(gpuClipScratchRcpW.data(), true);
			const auto tMocBuffer1 = std::chrono::steady_clock::now();
			mocBufferMs =
			    std::chrono::duration<double, std::milli>(tMocBuffer1 - tMocBuffer0).count();
		} else {
			const auto tMocBuffer0 = std::chrono::steady_clock::now();

			std::vector<size_t> order(objects.size());
			for (size_t i = 0; i < order.size(); ++i)
				order[i] = i;
			std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
				PackedVec3f midA{
				    (objects[a].mesh->aabbMin.x + objects[a].mesh->aabbMax.x) * 0.5f,
				    (objects[a].mesh->aabbMin.y + objects[a].mesh->aabbMax.y) * 0.5f,
				    (objects[a].mesh->aabbMin.z + objects[a].mesh->aabbMax.z) * 0.5f,
				};
				PackedVec3f midB{
				    (objects[b].mesh->aabbMin.x + objects[b].mesh->aabbMax.x) * 0.5f,
				    (objects[b].mesh->aabbMin.y + objects[b].mesh->aabbMax.y) * 0.5f,
				    (objects[b].mesh->aabbMin.z + objects[b].mesh->aabbMax.z) * 0.5f,
				};
				Vec3f ca3{midA.x, midA.y, midA.z};
				Vec3f cb3{midB.x, midB.y, midB.z};
				Vec4f ha = ca3 * objects[a].model;
				Vec4f hb = cb3 * objects[b].model;
				Vec3f ca{ha.x(), ha.y(), ha.z()};
				Vec3f cb{hb.x(), hb.y(), hb.z()};
				const float da = DepthKey(ca, camPosForSort);
				const float db = DepthKey(cb, camPosForSort);
				if (da < db)
					return true;
				if (db < da)
					return false;
				return a < b;
			});

			moc->ClearBuffer();
			for (size_t k : order) {
				const SceneObject &o = objects[k];
				const Mesh &mesh = *o.mesh;
				if (!MeshIsDrawable(mesh))
					continue;
				Matr4f mvp = o.model * vp;
				MrMatrToColumnMajorGl(mvp, mvpCol);
				std::vector<float> clipVerts(mesh.positions.size() * 4);
				MaskedOcclusionCulling::TransformVertices(
				    mvpCol,
				    &mesh.positions[0].x,
				    clipVerts.data(),
				    (unsigned)mesh.positions.size(),
				    MaskedOcclusionCulling::VertexLayout(12, 4, 8));
				moc->RenderTriangles(
				    clipVerts.data(),
				    mesh.indices.data(),
				    (int)(mesh.indices.size() / 3),
				    nullptr,
				    MaskedOcclusionCulling::BACKFACE_CW,
				    MaskedOcclusionCulling::CLIP_PLANE_ALL,
				    MaskedOcclusionCulling::VertexLayout(16, 4, 12));
			}
			const auto tMocBuffer1 = std::chrono::steady_clock::now();
			mocBufferMs =
			    std::chrono::duration<double, std::milli>(tMocBuffer1 - tMocBuffer0).count();
		}

		if (saveMocDepthFrame >= 0 && passFrameIndex == saveMocDepthFrame) {
			char path[96];
			std::snprintf(path, sizeof(path), "moc_depth%d.png", passFrameIndex);
			if (SaveMocHizDepthPng(moc, fbW, fbH, path)) {
				std::error_code ec;
				const std::filesystem::path abs =
				    std::filesystem::weakly_canonical(std::filesystem::path(path), ec);
				const std::string show =
				    ec ? std::filesystem::absolute(std::filesystem::path(path)).generic_string() : abs.generic_string();
				std::fprintf(stderr, "Wrote MOC hierarchical-z → per-pixel depth (tonemapped): %s\n", show.c_str());
			}
		}

		const unsigned nAll = (unsigned)objects.size();
		unsigned nFrustum = 0;
		unsigned nMocVisible = 0;
		unsigned nGpuVisible = 0;
		unsigned fp = 0, fn = 0;
		unsigned agreeVisFrustum = 0;

		std::vector<char> frustumHit(nAll, 0);
		std::vector<char> mocVis(nAll, 0);
		std::vector<char> mocTested(nAll, 0);
		std::vector<MaskedOcclusionCulling::CullingResult> mocRaw(nAll);
		std::vector<char> gpuVis(nAll, 0);

		Vec4f frustumPlanes[6];
		FrustumPlanesMrGraphics(vp, frustumPlanes);

		for (size_t i = 0; i < objects.size(); ++i) {
			const SceneObject &o = objects[i];
			Vec3f wcorners[8];
			ObjectWorldCorners(o, wcorners);
			bool inf = IsWorldAabbFrustumVisibleMrGraphics(frustumPlanes, wcorners);
			frustumHit[i] = inf ? 1 : 0;
			if (inf)
				++nFrustum;
		}

		const auto tMocQuery0 = std::chrono::steady_clock::now();
		for (size_t i = 0; i < objects.size(); ++i) {
			if (!frustumHit[i])
				continue;
			const SceneObject &o = objects[i];
			const Mesh &mesh = *o.mesh;
			if (!MeshIsDrawable(mesh))
				continue;
			Matr4f mvp = o.model * vp;
			MaskedOcclusionCulling::CullingResult r = MaskedOcclusionCulling::VIEW_CULLED;
			if (mocOccludeeTest == MocOccludeeTestMode::AabbScreenRect) {
				Vec3f wc[8];
				ObjectWorldCorners(o, wc);
				float rx0, ry0, rx1, ry1, rwMin;
				if (TryWorldAabbProjectToTestRect(wc, vp, rx0, ry0, rx1, ry1, rwMin))
					r = moc->TestRect(rx0, ry0, rx1, ry1, rwMin);
				else
					r = MaskedOcclusionCulling::VIEW_CULLED;
			} else if (mocOccludeeTest == MocOccludeeTestMode::AabbScreenTriangles) {
				Vec3f wc[8];
				ObjectWorldCorners(o, wc);
				std::vector<NdcPoint2f> ndcPoints;
				float wRef = FLT_MAX;
				bool usedTriangles = false;
				if (TryWorldAabbProjectToNdcPoints(wc, vp, ndcPoints, wRef)) {
					std::vector<NdcPoint2f> hull;
					if (BuildConvexHull2D(ndcPoints, hull)) {
						std::vector<float> clipVerts;
						std::vector<unsigned> triIdx;
						if (BuildClipFanFromHull(hull, wRef, clipVerts, triIdx)) {
							r = moc->TestTriangles(
							    clipVerts.data(),
							    triIdx.data(),
							    (int)(triIdx.size() / 3),
							    nullptr,
							    MaskedOcclusionCulling::BACKFACE_NONE,
							    MaskedOcclusionCulling::CLIP_PLANE_ALL,
							    MaskedOcclusionCulling::VertexLayout(16, 4, 12));
							usedTriangles = true;
						}
					}
				}
				if (!usedTriangles) {
					float rx0, ry0, rx1, ry1, rwMin;
					if (TryWorldAabbProjectToTestRect(wc, vp, rx0, ry0, rx1, ry1, rwMin))
						r = moc->TestRect(rx0, ry0, rx1, ry1, rwMin);
					else
						r = MaskedOcclusionCulling::VIEW_CULLED;
				}
			} else if (mocOccludeeTest == MocOccludeeTestMode::DopKScreenRect) {
				EnsureMeshDopK(mesh, mocDopK);
				const MeshDopLocal *dop = nullptr;
				auto dk = mesh.dopByK.find(mocDopK);
				if (dk != mesh.dopByK.end() && !dk->second.localVerts.empty())
					dop = &dk->second;
				float rx0, ry0, rx1, ry1, rwMin;
				bool got = false;
				if (!dop) {
					Vec3f wc[8];
					ObjectWorldCorners(o, wc);
					got = TryWorldAabbProjectToTestRect(wc, vp, rx0, ry0, rx1, ry1, rwMin);
				} else {
					got = TryWorldDop26ProjectToTestRect(
					    dop->localVerts, dop->edges, o.model, vp, rx0, ry0, rx1, ry1, rwMin);
					if (!got) {
						Vec3f wc[8];
						ObjectWorldCorners(o, wc);
						got = TryWorldAabbProjectToTestRect(wc, vp, rx0, ry0, rx1, ry1, rwMin);
					}
				}
				if (got)
					r = moc->TestRect(rx0, ry0, rx1, ry1, rwMin);
				else
					r = MaskedOcclusionCulling::VIEW_CULLED;
			} else if (mocOccludeeTest == MocOccludeeTestMode::DopKScreenTriangles) {
				EnsureMeshDopK(mesh, mocDopK);
				const MeshDopLocal *dop = nullptr;
				auto dk = mesh.dopByK.find(mocDopK);
				if (dk != mesh.dopByK.end() && !dk->second.localVerts.empty())
					dop = &dk->second;
				bool usedTriangles = false;
				if (dop) {
					std::vector<NdcPoint2f> ndcPoints;
					float wRef = FLT_MAX;
					if (TryWorldDop26ProjectToNdcPoints(
					        dop->localVerts, dop->edges, o.model, vp, ndcPoints, wRef)) {
						std::vector<NdcPoint2f> hull;
						if (BuildConvexHull2D(ndcPoints, hull)) {
							std::vector<float> clipVerts;
							std::vector<unsigned> triIdx;
							if (BuildClipFanFromHull(hull, wRef, clipVerts, triIdx)) {
								r = moc->TestTriangles(
								    clipVerts.data(),
								    triIdx.data(),
								    (int)(triIdx.size() / 3),
								    nullptr,
								    MaskedOcclusionCulling::BACKFACE_NONE,
								    MaskedOcclusionCulling::CLIP_PLANE_ALL,
								    MaskedOcclusionCulling::VertexLayout(16, 4, 12));
								usedTriangles = true;
							}
						}
					}
				}
				if (!usedTriangles) {
					float rx0, ry0, rx1, ry1, rwMin;
					bool got = false;
					if (!dop) {
						Vec3f wc[8];
						ObjectWorldCorners(o, wc);
						got = TryWorldAabbProjectToTestRect(wc, vp, rx0, ry0, rx1, ry1, rwMin);
					} else {
						got = TryWorldDop26ProjectToTestRect(
						    dop->localVerts, dop->edges, o.model, vp, rx0, ry0, rx1, ry1, rwMin);
						if (!got) {
							Vec3f wc[8];
							ObjectWorldCorners(o, wc);
							got = TryWorldAabbProjectToTestRect(wc, vp, rx0, ry0, rx1, ry1, rwMin);
						}
					}
					if (got)
						r = moc->TestRect(rx0, ry0, rx1, ry1, rwMin);
					else
						r = MaskedOcclusionCulling::VIEW_CULLED;
				}
			} else {
				MrMatrToColumnMajorGl(mvp, mvpCol);
				std::vector<float> clipVerts(mesh.positions.size() * 4);
				MaskedOcclusionCulling::TransformVertices(
				    mvpCol,
				    &mesh.positions[0].x,
				    clipVerts.data(),
				    (unsigned)mesh.positions.size(),
				    MaskedOcclusionCulling::VertexLayout(12, 4, 8));
				r = moc->TestTriangles(
				    clipVerts.data(),
				    mesh.indices.data(),
				    (int)(mesh.indices.size() / 3),
				    nullptr,
				    MaskedOcclusionCulling::BACKFACE_CW,
				    MaskedOcclusionCulling::CLIP_PLANE_ALL,
				    MaskedOcclusionCulling::VertexLayout(16, 4, 12));
			}
			mocTested[i] = 1;
			mocRaw[i] = r;
			if (r == MaskedOcclusionCulling::VISIBLE) {
				mocVis[i] = 1;
				++nMocVisible;
			}
		}
		const auto tMocQuery1 = std::chrono::steady_clock::now();
		const double mocQueryMs =
		    std::chrono::duration<double, std::milli>(tMocQuery1 - tMocQuery0).count();

		if (!usePrevFramePrev) {
			const auto tGpuRef0 = std::chrono::steady_clock::now();
			drawGpuReferenceFill(vp, gpuDrawAlways ? nullptr : &mocVis);
			glFinish();
			const auto tGpuDraw1 = std::chrono::steady_clock::now();
			gpuRefMs = std::chrono::duration<double, std::milli>(tGpuDraw1 - tGpuRef0).count();
		} else if (!gpuDrawAlways) {
			const auto tGpuCull0 = std::chrono::steady_clock::now();
			drawGpuReferenceFill(vp, &mocVis);
			glFinish();
			const auto tGpuCull1 = std::chrono::steady_clock::now();
			gpuRefMs += std::chrono::duration<double, std::milli>(tGpuCull1 - tGpuCull0).count();
		}

		std::vector<unsigned> pixels((size_t)fbW * (size_t)fbH * 4);
		const auto tReadPx0 = std::chrono::steady_clock::now();
		glReadBuffer(GL_COLOR_ATTACHMENT0);
		glReadPixels(0, 0, fbW, fbH, GL_RGBA_INTEGER, GL_UNSIGNED_INT, pixels.data());
		CheckGl("readpixels");
		const auto tReadPx1 = std::chrono::steady_clock::now();
		const double readPixelsMs =
		    std::chrono::duration<double, std::milli>(tReadPx1 - tReadPx0).count();

		std::unordered_set<uint32_t> uniq;
		for (size_t p = 0; p < pixels.size(); p += 4) {
			uint32_t id = pixels[p + 3];
			if (id != 0)
				uniq.insert(id);
		}

		for (size_t i = 0; i < objects.size(); ++i) {
			uint32_t id = objects[i].id;
			if (uniq.count(id))
				gpuVis[i] = 1;
		}

		/// Match nMocVisible domain: only instances that passed world-AABB frustum test (same loop as MOC queries).
		/// uniq.size() mixes in objects outside that AABB test but with rasterized pixels — bogus vs line 3).
		nGpuVisible = 0;
		for (size_t i = 0; i < objects.size(); ++i) {
			if (!frustumHit[i])
				continue;
			if (gpuVis[i])
				++nGpuVisible;
		}

		for (size_t i = 0; i < objects.size(); ++i) {
			if (!frustumHit[i])
				continue;
			if (mocVis[i] && gpuVis[i])
				++agreeVisFrustum;
			if (mocVis[i] && !gpuVis[i])
				++fp;
			if (!mocVis[i] && gpuVis[i])
				++fn;
		}

		const auto tPassWall1 = std::chrono::steady_clock::now();
		const double passWallMs =
		    std::chrono::duration<double, std::milli>(tPassWall1 - tPassWall0).count();

		AccBenchPassResult result;
		result.nAll = nAll;
		result.nFrustum = nFrustum;
		result.nMocVisible = nMocVisible;
		result.nGpuVisible = nGpuVisible;
		result.fp = fp;
		result.fn = fn;
		result.agreeVisFrustum = agreeVisFrustum;
		result.mocBufferMs = mocBufferMs;
		result.mocQueryMs = mocQueryMs;
		result.gpuRefMs = gpuRefMs;
		result.readPixelsMs = readPixelsMs;
		result.passWallMs = passWallMs;

		if (printStyle == BenchPrintStyle::Full) {
			Vec3f camPos = fps.cam().position();
			Vec3f camDir = Vec3f(fps.cam().direction());
			Vec3f camUp = Vec3f(fps.cam().up());
			Vec3f target = camPos + camDir;

			printf("\n");
			printf("=== How geometry is drawn (same MVP on CPU and GPU) ===\n");
			printf(
			    "  Mesh vertices are in object/local space (OBJ shape or unit cube).\n"
			    "  mr-math: p_clip_row = p_local * model * (view*proj); GL/MOC get column-major uMVP = that 4×4 (mr layout matches glm columns as rows).\n"
			    "  MOC: TransformVertices(MVP, local xyz) -> clip (x,y,z,w), then RenderTriangles /\n"
			    "       TestTriangles on that clip stream (see MaskedOcclusionCulling README: uses w, ~1/w depth).\n"
			    "  OpenGL: VBO = same local positions; vertex shader gl_Position = MVP * vec4(aPos,1).\n"
			    "          Fragment shader writes uvec4(0,0,0, objectId) to RGBA32UI; depth buffer GL_LESS.\n"
			    "  Frustum stats: same as mr-graphics — planes from viewProj().transposed() (vp[3]±vp[0..2]),\n"
			    "          normalized; world AABB = min/max of 8 local-AABB corners * model; bounds.h-style test;\n"
			    "          GPU reference draw + preview mesh skip draws when this test fails (same set as MOC queries);\n"
			    "          default submit for id readback = MOC VISIBLE only (`--gpu-draw-always` = full frustum overdraw ref).\n\n");
			printf("Camera pos (%.2f, %.2f, %.2f)  dir (%.2f, %.2f, %.2f)  up (%.2f, %.2f, %.2f)\n",
			    camPos.x(), camPos.y(), camPos.z(), camDir.x(), camDir.y(), camDir.z(), camUp.x(), camUp.y(),
			    camUp.z());
			printf("  -> look-at target (%.2f, %.2f, %.2f)   FOV 45° (mr Scene)   near %.2f  far %.2f\n",
			    target.x(), target.y(), target.z(), nearP, fps.cam().projection().far);

			if (perObjectReport) {
				printf("\n=== Per-object (translation column; id = GPU/MRT .a) ===\n");
				printf(
				    "%-4s %-5s %5s %5s %10s %10s %10s  %-7s  %-14s  %s\n",
				    "#", "id", "vtx", "tri", "tx", "ty", "tz", "frustum", "MOC", "GPU_pixel");
				for (size_t i = 0; i < objects.size(); ++i) {
					const SceneObject &o = objects[i];
					float tx = o.model[0][3];
					float ty = o.model[1][3];
					float tz = o.model[2][3];
					const char *fr = frustumHit[i] ? "yes" : "no";
					const char *mocS = mocTested[i] ? MocResultStr(mocRaw[i]) : "—";
					const char *gpuS = gpuVis[i] ? "yes" : "no";
					printf(
					    "%-4zu %-5u %5zu %5zu %10.2f %10.2f %10.2f  %-7s  %-14s  %s\n",
					    i, o.id, o.mesh->positions.size(), o.mesh->indices.size() / 3, tx, ty, tz, fr, mocS, gpuS);
				}
			}

			printf("\n--- AccuracyBench (MaskedOcclusionCulling vs RGBA32UI id in .a) ---\n");
			printf(
			    "Resolution %dx%d  MOC USE_D3D=%d (see MaskedOcclusionCulling.h)  near=%.2f  "
			    "moc-test=%s (ACCURACYBENCH_MOC_TEST or --moc-test=)\n",
			    fbW, fbH, USE_D3D, nearP, MocTestModeLabel(mocOccludeeTest, mocDopK).c_str());
			printf("1) All objects:              %u\n", result.nAll);
			printf("2) AABB in frustum:          %u\n", result.nFrustum);
			printf("3) MOC VISIBLE (occludee test; frustum subset): %u\n", result.nMocVisible);
			printf(
			    "4) GPU-visible instances (frustum AABB subset, ≥1 pixel in id readback): %u  "
			    "(global unique ids in buffer — reference only: %zu)%s\n",
			    result.nGpuVisible,
			    uniq.size(),
			    gpuDrawAlways ? "" : "  [default: culled draw — only MOC VISIBLE submitted]");
			if (gpuDrawAlways) {
				const int lhs = (int)result.nGpuVisible - (int)result.nMocVisible;
				const int rhs = (int)result.fn - (int)result.fp;
				printf(
				    "4b) Sanity: gpuIds−mocVis should equal FN−FP → %d vs %d %s\n",
				    lhs, rhs, lhs == rhs ? "(ok)" : "(BUG)");
				printf(
				    "4c) Frustum ∩ both visible: %u  (= mocVis−FP = gpuIds−FN → %u = %u)\n",
				    result.agreeVisFrustum,
				    result.nMocVisible - result.fp,
				    result.nGpuVisible - result.fn);
			} else {
				printf(
				    "4b) Default culled draw: occluded instances not rasterized → FN pixel metric unavailable; "
				    "expect FN≈0, gpuIds≤mocVis if visibility gate matches.\n");
			}
			if (gpuDrawAlways && mocOccludeeTest != MocOccludeeTestMode::Mesh) {
				printf(
				    "    Proxy --moc-test: line 3 uses occludee proxy; line 4 uses mesh pixels. "
				    "dop/AABB hull ⊂ mesh ⇒ gpuIds>mocVis normal (see FN).\n");
			}
			printf(
			    "5) MOC hierarchical-Z build (%s): %.3f ms\n",
			    usePrevFramePrev
			        ? (gpuDrawAlways
			              ? "readback clip.w + Import (~1/w); same pass as timed GPU draw in 7)"
			              : "readback clip.w from pass1 (all frustum) + ImportPixelDepthBuffer (~1/w)")
			        : "clear + sort + RenderTriangles",
			    result.mocBufferMs);
			printf(
			    "6) MOC occlusion queries (%s): %.3f ms\n",
			    mocOccludeeTest == MocOccludeeTestMode::AabbScreenRect
			        ? "TestRect from world AABB (edges clipped to w=min clip plane)"
			    : mocOccludeeTest == MocOccludeeTestMode::AabbScreenTriangles
			        ? "World AABB to NDC; convex hull 2D fan via TestTriangles (fallback TestRect)"
			    : mocOccludeeTest == MocOccludeeTestMode::DopKScreenRect
			        ? "TestRect from k-dop in NDC (w-clip on edges) or AABB if no DOP / degenerate"
			    : mocOccludeeTest == MocOccludeeTestMode::DopKScreenTriangles
			        ? "Projected k-dop to NDC; convex hull fan via TestTriangles (fallback TestRect)"
			        : "TransformVertices + TestTriangles per object",
			    result.mocQueryMs);
			printf(
			    !gpuDrawAlways && usePrevFramePrev
			        ? "7) GPU draw (pass1 all frustum→clip.w Import + pass2 culled color/depth): %.3f ms\n"
			    : !gpuDrawAlways
			        ? "7) GPU draw (after queries: culled visible-only submit): %.3f ms\n"
			    : usePrevFramePrev
			        ? "7) GPU draw (single ref pass — shared id buffer + clip.w for Import): %.3f ms\n"
			        : "7) GPU draw (FBO clear + buffer upload + draw + glFinish): %.3f ms\n",
			    result.gpuRefMs);
			printf("8) glReadPixels (RGBA32UI readback for benchmark): %.3f ms\n", result.readPixelsMs);
			printf(
			    "9) Full benchmark pass wall (Create → FP/FN stats; incl. frustum + pixel scan): %.3f ms\n",
			    result.passWallMs);
			printf("10) Sum of 5)+6)+7)+8) (sequential stages): %.3f ms\n",
			    result.mocBufferMs + result.mocQueryMs + result.gpuRefMs + result.readPixelsMs);
			printf("--- errors (frustum subset only; GPU ref = mesh, MOC = --moc-test geometry) ---\n");
			printf("False positives (MOC visible, GPU no pixel): %u\n", result.fp);
			printf("False negatives (MOC occluded/culled, GPU pixel): %u\n", result.fn);
			if (!gpuDrawAlways)
				printf(
				    "(Default culled draw: FN column ~meaningless — occluded objects never submitted to GPU ref pass.)\n");
			else if (mocOccludeeTest == MocOccludeeTestMode::Mesh)
				printf(
				    "(Mesh mode: conservative test vs same raster — FN>0 ⇒ Hi-Z/import/projection mismatch worth chasing.)\n");
			else
				printf(
				    "(Proxy mode: FN counts mesh outside proxy footprint in Hi-Z test — expected if dop/AABB hull tighter than silhouette.)\n");
			fflush(stdout);
		}

		/// Same one-line summary as the interactive preview loop (stderr), for headless --max-frames= batching.
		/// No GUI in this path; 10)guiFrame= is 0.0. Omit when not headless so the window path does not double-print.
		if (headless && maxFramesLimit > 0) {
			const double guiFrameMs = 0.0;
			const unsigned agr = result.agreeVisFrustum;
			const bool inv =
			    (agr + result.fp == result.nMocVisible) && (agr + result.fn == result.nGpuVisible) &&
			    ((int)result.nGpuVisible - (int)result.nMocVisible == (int)result.fn - (int)result.fp);
			std::fprintf(stderr,
			    "ACCBench  1)all=%u  2)frustum=%u  3)mocVis=%u  4)gpuIds=%u  agr=%u fp=%u fn=%u  stat=%s  "
			    "5)mocBuf=%.3fms  6)mocQry=%.3fms  7)gpuDraw=%.3fms  8)readPx=%.3fms  9)passWall=%.3fms  "
			    "10)guiFrame=%.3fms\n",
			    result.nAll, result.nFrustum, result.nMocVisible, result.nGpuVisible, agr, result.fp, result.fn,
			    inv ? "ok" : "BUG", result.mocBufferMs, result.mocQueryMs, result.gpuRefMs, result.readPixelsMs,
			    result.passWallMs, guiFrameMs);
			std::fflush(stderr);
		}

		for (size_t pi = 0; pi < objects.size(); ++pi)
			guiBenchMocVisible[pi] = static_cast<unsigned char>(mocVis[pi]);

		// Default framebuffer must be bound for GUI preview paths; detach VAO/program so attrib 0 /
		// draw-buffer state cannot leak after RGBA32UI FBO draws + readback.
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glBindVertexArray(0);
		glUseProgram(0);

		MaskedOcclusionCulling::Destroy(moc);
		return result;
	};

	if (saveMocDepthFrame >= 0) {
		std::error_code ecCwd;
		const std::filesystem::path cwd = std::filesystem::current_path(ecCwd);
		std::fprintf(stderr, "ACCURACYBENCH: --save-depth-frame → PNG written under cwd: %s\n",
		    ecCwd ? "(current_path failed)" : cwd.generic_string().c_str());
	}

	Matr4f vpBench = fps.viewProj();
	Vec3f camPosBench = fps.cam().position();
	/// Batch N passes: always when headless+--max-frames; also when --save-depth-frame set (so dump
	/// works without --headless — otherwise only pass 0 runs before GUI, frame K needs K+1 swaps).
	if (maxFramesLimit > 0 && (headless || saveMocDepthFrame >= 0)) {
		if (saveMocDepthFrame >= maxFramesLimit) {
			std::fprintf(stderr,
			    "ACCURACYBENCH: --save-depth-frame=%d >= --max-frames=%d (valid indices 0..%d); no save.\n",
			    saveMocDepthFrame, maxFramesLimit, maxFramesLimit - 1);
		}
		for (int f = 0; f < maxFramesLimit; ++f) {
			Matr4f vpB = fps.viewProj();
			Vec3f cpos = fps.cam().position();
			runBenchmarkPass(f, vpB, cpos, f == 0 ? BenchPrintStyle::Full : BenchPrintStyle::LiveFour);
		}
	} else
		runBenchmarkPass(0, vpBench, camPosBench, BenchPrintStyle::Full);

	if (!headless) {
		printf("\nInteractive preview: WASD + Space/Z, Shift sprint, mouse look, scroll = move speed; "
		       "P = print pose; 1–6,0 = scene camera presets.\n");
		printf(
		    "With `--moc-test` other than mesh: `--visualize-bounds` draws k-DOP / AABB wires (green; depth-tested, "
		    "no depth writes). `--visualize-bound-projection` draws NDC hull (yellow); alias `--visualize-bounds-projection`. "
		    "Both only for objects MOC-visible this frame or last frame.\n");
		printf("Live stats on stderr: counts + MOC build / query + GPU draw + readPixels + pass wall + "
		       "full GUI frame (ms), every iteration.\n");
		if (maxFramesLimit > 0)
			printf("Exiting after %d preview frame(s) (--max-frames); you can also close the window to quit early.\n",
			    maxFramesLimit);
		else
			printf("Close the window to quit.\n");
		fflush(stdout);
	}

	if (!headless && mocOccludeeTest == MocOccludeeTestMode::Mesh &&
	    (visualizeBounds || visualizeBoundProjection)) {
		std::fprintf(stderr,
		    "ACCURACYBENCH: --visualize-bounds / --visualize-bound-projection require `--moc-test` other than "
		    "mesh; no bound overlay will be drawn.\n");
	}

	if (!headless) {
		double lx = 0, ly = 0;
		bool haveCursor = false;
		int frameI = 0;
		while (!glfwWindowShouldClose(win)) {
			glfwPollEvents();

			int winW = fbW, winH = fbH;
			glfwGetFramebufferSize(win, &winW, &winH);
			// Projection matches letterboxed vw:vh ≡ fbW:fbH, not raw win buffer (scaling → VP/viewport mismatch).
			fps.configureProjection(aspect0, clipNear, clipFar);

			double cx, cy;
			glfwGetCursorPos(win, &cx, &cy);
			float dx = 0.f, dy = 0.f;
			if (haveCursor) {
				dx = float(cx - lx);
				dy = float(cy - ly);
				// Ignore huge jumps (focus loss, first grab) so the view doesn't flip.
				const float maxStep = 240.f;
				if (std::fabs(dx) > maxStep || std::fabs(dy) > maxStep)
					dx = dy = 0.f;
			}
			haveCursor = true;
			lx = cx;
			ly = cy;

			float minSp = 0.005f, maxSp = 20.f;
			float spCoef = 0.005f;
			float newSp = fps.speed() + float(g_scrollAccum) * spCoef;
			g_scrollAccum = 0.0;
			if (newSp >= minSp && newSp <= maxSp)
				fps.speed(newSp);

			// Same normalization as mr::graphics Scene::update (mouse delta / extent, minus Y for screen space).
			Vec3f angularDelta{dx / float(std::max(winW, 1)), -dy / float(std::max(winH, 1)), 0.f};
			fps.turn(angularDelta);

			float speedup = glfwGetKey(win, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ? 10.f : 1.f;
			if (glfwGetKey(win, GLFW_KEY_W) == GLFW_PRESS)
				fps.move(Vec3f(fps.cam().direction()) * speedup);
			if (glfwGetKey(win, GLFW_KEY_A) == GLFW_PRESS)
				fps.move(-Vec3f(fps.cam().right()) * speedup);
			if (glfwGetKey(win, GLFW_KEY_S) == GLFW_PRESS)
				fps.move(-Vec3f(fps.cam().direction()) * speedup);
			if (glfwGetKey(win, GLFW_KEY_D) == GLFW_PRESS)
				fps.move(Vec3f(fps.cam().right()) * speedup);
			if (glfwGetKey(win, GLFW_KEY_SPACE) == GLFW_PRESS)
				fps.move(Vec3f(fps.cam().up()) * speedup);
			if (glfwGetKey(win, GLFW_KEY_Z) == GLFW_PRESS)
				fps.move(-Vec3f(fps.cam().up()) * speedup);

			static bool k1, k2, k3, k4, k5, k6, k0, kEsc, kP;
			auto edge = [&](int key, bool &prev) {
				bool now = glfwGetKey(win, key) == GLFW_PRESS;
				bool e = now && !prev;
				prev = now;
				return e;
			};
			if (edge(GLFW_KEY_P, kP)) {
				Vec3f p = fps.cam().position();
				Vec3f d = Vec3f(fps.cam().direction());
				Vec3f u = Vec3f(fps.cam().up());
				printf("camera pos (%.3f, %.3f, %.3f) dir (%.3f, %.3f, %.3f) up (%.3f, %.3f, %.3f)\n",
				    p.x(), p.y(), p.z(), d.x(), d.y(), d.z(), u.x(), u.y(), u.z());
			}
			if (edge(GLFW_KEY_1, k1))
				fps.cam() = mr::math::Camera<float>(Vec3f{1, 1, 1}, Norm3f{-1, -1, -1}, Norm3f{0, 1, 0});
			if (edge(GLFW_KEY_2, k2))
				fps.cam() = mr::math::Camera<float>(Vec3f{10, 10, 10}, Norm3f{-1, -1, -1}, Norm3f{0, 1, 0});
			if (edge(GLFW_KEY_3, k3))
				fps.cam() = mr::math::Camera<float>(Vec3f{100, 100, 100}, Norm3f{-1, -1, -1}, Norm3f{0, 1, 0});
			if (edge(GLFW_KEY_4, k4))
				fps.cam() = mr::math::Camera<float>(Vec3f{500, 500, 500}, Norm3f{-1, -1, -1}, Norm3f{0, 1, 0});
			if (edge(GLFW_KEY_5, k5))
				fps.cam() =
				    mr::math::Camera<float>(Vec3f{10000, 10000, 10000}, Norm3f{-1, -1, -1}, Norm3f{0, 1, 0});
			if (edge(GLFW_KEY_6, k6))
				fps.cam() = mr::math::Camera<float>(
				    Vec3f{100000, 100000, 100000}, Norm3f{-1, -1, -1}, Norm3f{0, 1, 0});
			if (edge(GLFW_KEY_0, k0)) {
				Vec3f cp = fps.cam().position();
				auto nd = cp.normalized();
				if (nd)
					fps.cam() = mr::math::Camera<float>(cp, -(*nd), Norm3f{0, 1, 0});
			}
			if (edge(GLFW_KEY_ESCAPE, kEsc))
				glfwSetWindowShouldClose(win, GLFW_TRUE);

			Matr4f vp = fps.viewProj();
			const auto tGuiFrame0 = std::chrono::steady_clock::now();
			AccBenchPassResult br =
			    runBenchmarkPass(frameI, vp, fps.cam().position(), BenchPrintStyle::LiveFour);
			float mvpCol[16];

			glBindFramebuffer(GL_FRAMEBUFFER, 0);
			glViewport(0, 0, winW, winH);
			glClearColor(0.02f, 0.02f, 0.03f, 1.f);
			glClear(GL_COLOR_BUFFER_BIT);
			const float scale = std::min(winW / float(fbW), winH / float(fbH));
			const int vw = std::max(1, (int)(fbW * scale + 0.5f));
			const int vh = std::max(1, (int)(fbH * scale + 0.5f));
			const int ox = (winW - vw) / 2;
			const int oy = (winH - vh) / 2;
			glViewport(ox, oy, vw, vh);
			glEnable(GL_SCISSOR_TEST);
			glScissor(ox, oy, vw, vh);
			glClearColor(0.06f, 0.07f, 0.09f, 1.f);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			glDisable(GL_SCISSOR_TEST);

			glEnable(GL_DEPTH_TEST);
			glDepthFunc(GL_LESS);
			glUseProgram(progPreview);
			glBindVertexArray(vao);
			Vec4f previewFrustum[6];
			FrustumPlanesMrGraphics(vp, previewFrustum);
			for (size_t i = 0; i < objects.size(); ++i) {
				const SceneObject &o = objects[i];
				if (!MeshIsDrawable(*o.mesh))
					continue;
				Vec3f wcPreview[8];
				ObjectWorldCorners(o, wcPreview);
				if (!IsWorldAabbFrustumVisibleMrGraphics(previewFrustum, wcPreview))
					continue;
				if (!gpuDrawAlways && !guiBenchMocVisible[i])
					continue;
				Matr4f mvp = o.model * vp;
				MrMatrToColumnMajorGl(mvp, mvpCol);
				glUniformMatrix4fv(locMvpPrev, 1, GL_FALSE, mvpCol);
				glUniform1ui(locIdPrev, o.id);
				glBindBuffer(GL_ARRAY_BUFFER, vbo);
				glBufferData(GL_ARRAY_BUFFER, o.mesh->positions.size() * sizeof(PackedVec3f),
				    o.mesh->positions.data(), GL_STREAM_DRAW);
				glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
				glBufferData(GL_ELEMENT_ARRAY_BUFFER, o.mesh->indices.size() * sizeof(unsigned),
				    o.mesh->indices.data(), GL_STREAM_DRAW);
				glEnableVertexAttribArray(0);
				glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(PackedVec3f), (void *)0);
				glDrawElements(GL_TRIANGLES, (GLsizei)o.mesh->indices.size(), GL_UNSIGNED_INT, nullptr);
			}
			CheckGl("preview draw");

			if ((visualizeBounds || visualizeBoundProjection) && mocOccludeeTest != MocOccludeeTestMode::Mesh) {
				Vec4f frustumPlanes[6];
				FrustumPlanesMrGraphics(vp, frustumPlanes);
				std::vector<Vec3f> worldPairs;
				std::vector<float> wf;
				std::vector<float> nf;
				bool hasNdc = false;
				bool ndcIsRect = false;
				float rx0 = 0.f, ry0 = 0.f, rx1 = 0.f, ry1 = 0.f;
				std::vector<NdcPoint2f> ndcHull;

				glDisable(GL_BLEND);

				if (visualizeBounds) {
					glUseProgram(progLineWorld);
					MrMatrToColumnMajorGl(vp, mvpCol);
					glUniformMatrix4fv(locLineMvp, 1, GL_FALSE, mvpCol);
					glBindVertexArray(vao);
					glBindBuffer(GL_ARRAY_BUFFER, lineVbo);
					glDisable(GL_CULL_FACE);
					// glLineWidth is not guaranteed in Core profile; harmless if ignored.
					glLineWidth(2.0f);
					// Depth test + no writes: nearer mesh shading kept; omit solid bbox fill —
					// dense instances: unrelated AABB tris win depth-per-pixel falsely -> greenwash.
					glEnable(GL_DEPTH_TEST);
					glDepthFunc(GL_LESS);
					glDepthMask(GL_FALSE);

					for (size_t ii = 0; ii < objects.size(); ++ii) {
						if (!MeshIsDrawable(*objects[ii].mesh))
							continue;
						if (!guiBenchMocVisible[ii] && !guiPrevMocVisibleForOverlay[ii])
							continue;
						Vec3f wcForFrust[8];
						ObjectWorldCorners(objects[ii], wcForFrust);
						if (!IsWorldAabbFrustumVisibleMrGraphics(frustumPlanes, wcForFrust))
							continue;

						FillOccludeeBoundOverlay(objects[ii], *objects[ii].mesh, mocOccludeeTest, mocDopK, vp,
						    true, false, worldPairs, hasNdc, ndcIsRect, rx0, ry0, rx1, ry1, ndcHull);
						if (!worldPairs.empty()) {
							glUniform3f(locLineColor, 0.28f, 0.92f, 0.42f);
							wf.resize(worldPairs.size() * 3);
							for (size_t j = 0; j < worldPairs.size(); ++j) {
								wf[j * 3 + 0] = worldPairs[j].x();
								wf[j * 3 + 1] = worldPairs[j].y();
								wf[j * 3 + 2] = worldPairs[j].z();
							}
							glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(wf.size() * sizeof(float)), wf.data(),
							    GL_STREAM_DRAW);
							glEnableVertexAttribArray(0);
							glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, (void *)0);
							glDrawArrays(GL_LINES, 0, (GLsizei)(wf.size() / 3));
						}
					}
					CheckGl("preview bound wires");
				}
				if (visualizeBoundProjection) {
					glBindFramebuffer(GL_FRAMEBUFFER, 0);
					glDisable(GL_DEPTH_TEST);
					glDisable(GL_CULL_FACE);
					glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
					glEnable(GL_BLEND);
					glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
					glUseProgram(progOverlayNdc);
					glUniform3f(locNdcColor, 1.0f, 0.88f, 0.12f);
					glBindVertexArray(vao);
					glBindBuffer(GL_ARRAY_BUFFER, lineVbo);
					glLineWidth(2.0f);
					for (size_t ii = 0; ii < objects.size(); ++ii) {
						if (!MeshIsDrawable(*objects[ii].mesh))
							continue;
						if (!guiBenchMocVisible[ii] && !guiPrevMocVisibleForOverlay[ii])
							continue;
						Vec3f wcForFrust[8];
						ObjectWorldCorners(objects[ii], wcForFrust);
						if (!IsWorldAabbFrustumVisibleMrGraphics(frustumPlanes, wcForFrust))
							continue;
						FillOccludeeBoundOverlay(objects[ii], *objects[ii].mesh, mocOccludeeTest, mocDopK, vp,
						    false, true, worldPairs, hasNdc, ndcIsRect, rx0, ry0, rx1, ry1, ndcHull);
						if (!hasNdc)
							continue;
						glEnableVertexAttribArray(0);
						glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, (void *)0);
						if (ndcIsRect) {
							glUniform1f(locNdcAlpha, 0.22f);
							nf = {rx0, ry0, rx1, ry0, rx0, ry1, rx1, ry1};
							glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(nf.size() * sizeof(float)), nf.data(),
							    GL_STREAM_DRAW);
							glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
							glUniform1f(locNdcAlpha, 1.0f);
							nf = {
							    rx0, ry0, rx1, ry0,
							    rx1, ry0, rx1, ry1,
							    rx1, ry1, rx0, ry1,
							    rx0, ry1, rx0, ry0,
							};
							glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(nf.size() * sizeof(float)), nf.data(),
							    GL_STREAM_DRAW);
							glDrawArrays(GL_LINES, 0, 8);
						} else if (ndcHull.size() >= 3) {
							const size_t hn = ndcHull.size();
							glUniform1f(locNdcAlpha, 0.22f);
							nf.resize(hn * 2);
							for (size_t j = 0; j < hn; ++j) {
								nf[j * 2] = ndcHull[j].x;
								nf[j * 2 + 1] = ndcHull[j].y;
							}
							glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(nf.size() * sizeof(float)), nf.data(),
							    GL_STREAM_DRAW);
							glDrawArrays(GL_TRIANGLE_FAN, 0, (GLsizei)hn);

							glUniform1f(locNdcAlpha, 1.0f);
							nf.resize(hn * 4);
							for (size_t j = 0; j < hn; ++j) {
								size_t jn = (j + 1) % hn;
								nf[j * 4 + 0] = ndcHull[j].x;
								nf[j * 4 + 1] = ndcHull[j].y;
								nf[j * 4 + 2] = ndcHull[jn].x;
								nf[j * 4 + 3] = ndcHull[jn].y;
							}
							glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(nf.size() * sizeof(float)), nf.data(),
							    GL_STREAM_DRAW);
							glDrawArrays(GL_LINES, 0, (GLsizei)(hn * 2));
						}
					}
					glDisable(GL_BLEND);
					CheckGl("preview bound NDC overlay");
				}
				glEnable(GL_DEPTH_TEST);
				glDepthMask(GL_TRUE);
			}
			glfwSwapBuffers(win);
			const auto tGuiFrame1 = std::chrono::steady_clock::now();
			const double guiFrameMs =
			    std::chrono::duration<double, std::milli>(tGuiFrame1 - tGuiFrame0).count();
			{
				const unsigned agr = br.agreeVisFrustum;
				const bool inv = (agr + br.fp == br.nMocVisible) && (agr + br.fn == br.nGpuVisible) &&
				    ((int)br.nGpuVisible - (int)br.nMocVisible == (int)br.fn - (int)br.fp);
				std::fprintf(stderr,
				    "ACCBench  1)all=%u  2)frustum=%u  3)mocVis=%u  4)gpuIds=%u  agr=%u fp=%u fn=%u  stat=%s  "
				    "5)mocBuf=%.3fms  6)mocQry=%.3fms  7)gpuDraw=%.3fms  8)readPx=%.3fms  9)passWall=%.3fms  "
				    "10)guiFrame=%.3fms\n",
				    br.nAll, br.nFrustum, br.nMocVisible, br.nGpuVisible, agr, br.fp, br.fn, inv ? "ok" : "BUG",
				    br.mocBufferMs, br.mocQueryMs, br.gpuRefMs, br.readPixelsMs, br.passWallMs, guiFrameMs);
			}
			std::fflush(stderr);
			guiPrevMocVisibleForOverlay.assign(guiBenchMocVisible.begin(), guiBenchMocVisible.end());
			++frameI;
			if (maxFramesLimit > 0 && frameI >= maxFramesLimit)
				break;
		}
		std::fprintf(stderr, "\n");
	}

	glDeleteRenderbuffers(1, &depthRb);
	glDeleteTextures(1, &colorTex);
	glDeleteTextures(1, &clipWTex);
	glDeleteFramebuffers(1, &fbo);
	glDeleteBuffers(1, &vbo);
	glDeleteBuffers(1, &ebo);
	glDeleteBuffers(1, &lineVbo);
	glDeleteVertexArrays(1, &vao);
	glDeleteProgram(progOverlayNdc);
	glDeleteProgram(progLineWorld);
	glDeleteProgram(progPreview);
	glDeleteProgram(prog);
	glfwDestroyWindow(win);
	glfwTerminate();
	return 0;
}
