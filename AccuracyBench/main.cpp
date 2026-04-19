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
#include <cstdlib>
#include <cstdio>
#include <cstddef>
#include <cstdlib>
#include <cstring>
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
#include "usd_load.hpp"
#if ACCBENCH_HAVE_MR_IMPORTER
#include "mr_import_usd.hpp"
#endif

#include "../MaskedOcclusionCulling.h"

namespace {

using mr::Matr4f;
using mr::PackedVec3f;
using mr::Vec3f;
using mr::Vec4f;
using mr::math::Camera;

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

const char *kFragSrc = R"(#version 330 core
uniform uint uObjectId;
layout(location = 0) out uvec4 mrt;
void main() {
	mrt = uvec4(0u, 0u, 0u, uObjectId);
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

struct Mesh {
	std::vector<PackedVec3f> positions;
	std::vector<unsigned int> indices;
	PackedVec3f aabbMin{};
	PackedVec3f aabbMax{};
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

int main(int argc, char **argv) {
	bool headless = false;
	bool perObjectReport = false;
	const char *objPath = nullptr;
	const char *cameraSpec = nullptr;
	int fbW = 1280;
	int fbH = 720;
	float clipNear = 0.01f;
	float clipFar = 1000.f;
	for (int i = 1; i < argc; ++i) {
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
		if (argv[i][0] != '-' && !objPath)
			objPath = argv[i];
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
		fps.cam() = mr::math::Camera<float>(cpos, cdir, cup);
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

	GLuint vs = CompileShader(GL_VERTEX_SHADER, kVertSrc);
	GLuint fs = CompileShader(GL_FRAGMENT_SHADER, kFragSrc);
	GLuint prog = LinkProgram(vs, fs);
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

	GLuint vao = 0, vbo = 0, ebo = 0;
	glGenVertexArrays(1, &vao);
	glGenBuffers(1, &vbo);
	glGenBuffers(1, &ebo);

	GLuint fbo = 0, colorTex = 0, depthRb = 0;
	glGenFramebuffers(1, &fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	glGenTextures(1, &colorTex);
	glBindTexture(GL_TEXTURE_2D, colorTex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32UI, fbW, fbH, 0, GL_RGBA_INTEGER, GL_UNSIGNED_INT, nullptr);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorTex, 0);
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

	auto runBenchmarkPass = [&](const Matr4f &vp, const Vec3f &camPosForSort, BenchPrintStyle printStyle) {
		MaskedOcclusionCulling *moc = MaskedOcclusionCulling::Create(MocImplFromEnv());
		moc->SetResolution((unsigned)fbW, (unsigned)fbH);
		moc->SetNearClipPlane(fps.cam().projection().distance);
		const auto tMocBuffer0 = std::chrono::steady_clock::now();
		moc->ClearBuffer();

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

		float mvpCol[16];
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
		const double mocBufferMs =
		    std::chrono::duration<double, std::milli>(tMocBuffer1 - tMocBuffer0).count();

		const unsigned nAll = (unsigned)objects.size();
		unsigned nFrustum = 0;
		unsigned nMocVisible = 0;
		unsigned nGpuVisible = 0;
		unsigned fp = 0, fn = 0;

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
			MrMatrToColumnMajorGl(mvp, mvpCol);
			std::vector<float> clipVerts(mesh.positions.size() * 4);
			MaskedOcclusionCulling::TransformVertices(
			    mvpCol,
			    &mesh.positions[0].x,
			    clipVerts.data(),
			    (unsigned)mesh.positions.size(),
			    MaskedOcclusionCulling::VertexLayout(12, 4, 8));
			MaskedOcclusionCulling::CullingResult r = moc->TestTriangles(
			    clipVerts.data(),
			    mesh.indices.data(),
			    (int)(mesh.indices.size() / 3),
			    nullptr,
			    MaskedOcclusionCulling::BACKFACE_CW,
			    MaskedOcclusionCulling::CLIP_PLANE_ALL,
			    MaskedOcclusionCulling::VertexLayout(16, 4, 12));
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

		glViewport(0, 0, fbW, fbH);
		glBindFramebuffer(GL_FRAMEBUFFER, fbo);
		const GLuint clearZ[4] = {0, 0, 0, 0};
		glClearBufferuiv(GL_COLOR, 0, clearZ);
		glClearDepth(1.0);
		glClear(GL_DEPTH_BUFFER_BIT);
		glEnable(GL_DEPTH_TEST);
		glDepthFunc(GL_LESS);
		glUseProgram(prog);
		glBindVertexArray(vao);

		for (size_t i = 0; i < objects.size(); ++i) {
			const SceneObject &o = objects[i];
			if (!MeshIsDrawable(*o.mesh))
				continue;
			Matr4f mvp = o.model * vp;
			MrMatrToColumnMajorGl(mvp, mvpCol);
			glUniformMatrix4fv(locMvp, 1, GL_FALSE, mvpCol);
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
		CheckGl("draw");
		glFinish();

		std::vector<unsigned> pixels((size_t)fbW * (size_t)fbH * 4);
		glReadBuffer(GL_COLOR_ATTACHMENT0);
		glReadPixels(0, 0, fbW, fbH, GL_RGBA_INTEGER, GL_UNSIGNED_INT, pixels.data());
		CheckGl("readpixels");

		std::unordered_set<uint32_t> uniq;
		for (size_t p = 0; p < pixels.size(); p += 4) {
			uint32_t id = pixels[p + 3];
			if (id != 0)
				uniq.insert(id);
		}
		nGpuVisible = (unsigned)uniq.size();

		for (size_t i = 0; i < objects.size(); ++i) {
			uint32_t id = objects[i].id;
			if (uniq.count(id))
				gpuVis[i] = 1;
		}

		for (size_t i = 0; i < objects.size(); ++i) {
			if (!frustumHit[i])
				continue;
			if (mocVis[i] && !gpuVis[i])
				++fp;
			if (!mocVis[i] && gpuVis[i])
				++fn;
		}

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
			    "          normalized; world AABB = min/max of 8 local-AABB corners * model; bounds.h-style test.\n\n");
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
			printf("Resolution %dx%d  MOC USE_D3D=%d (see MaskedOcclusionCulling.h)  near=%.2f\n",
			    fbW, fbH, USE_D3D, nearP);
			printf("1) All objects:              %u\n", nAll);
			printf("2) AABB in frustum:          %u\n", nFrustum);
			printf("3) MOC TestTriangles VISIBLE (subset of frustum): %u\n", nMocVisible);
			printf("4) GPU unique object IDs in buffer (any pixel):   %u\n", nGpuVisible);
			printf(
			    "5) MOC hierarchical-Z build (clear + sort + RenderTriangles): %.3f ms\n",
			    mocBufferMs);
			printf(
			    "6) MOC occlusion queries (TransformVertices + TestTriangles): %.3f ms\n",
			    mocQueryMs);
			printf("--- errors (frustum subset only) ---\n");
			printf("False positives (MOC visible, GPU no pixel): %u\n", fp);
			printf("False negatives (MOC occluded/culled, GPU pixel): %u\n", fn);
			printf("(Conservative culling should avoid false negatives; a non-zero count means mismatch.)\n");
			fflush(stdout);
		} else {
			std::fprintf(stderr,
			    "\rACCBench  1)all=%u  2)frustum=%u  3)mocVis=%u  4)gpuIds=%u  "
			    "5)mocBuf=%.3fms  6)mocQry=%.3fms\033[K",
			    nAll, nFrustum, nMocVisible, nGpuVisible, mocBufferMs, mocQueryMs);
			std::fflush(stderr);
		}

		MaskedOcclusionCulling::Destroy(moc);
	};

	Matr4f vpBench = fps.viewProj();
	Vec3f camPosBench = fps.cam().position();
	runBenchmarkPass(vpBench, camPosBench, BenchPrintStyle::Full);

	if (!headless) {
		printf("\nInteractive preview: WASD + Space/Z, Shift sprint, mouse look, scroll = move speed; "
		       "P = print pose; 1–6,0 = scene camera presets.\n");
		printf("Live stats on stderr: 1)–4) counts + 5)–6) MOC buffer / query CPU time (ms), every frame "
		       "(full MOC+GPU readback pass).\n");
		printf("Close the window to quit.\n");
		fflush(stdout);
	}

	if (!headless) {
		double lx = 0, ly = 0;
		bool haveCursor = false;
		while (!glfwWindowShouldClose(win)) {
			glfwPollEvents();

			int winW = fbW, winH = fbH;
			glfwGetFramebufferSize(win, &winW, &winH);
			const float asp = winH > 0 ? float(winW) / float(winH) : aspect0;
			fps.configureProjection(asp, clipNear, clipFar);

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
				fps.cam() = mr::math::Camera<float>(Vec3f{1, 1, 1}, Vec3f{-1, -1, -1}, Vec3f{0, 1, 0});
			if (edge(GLFW_KEY_2, k2))
				fps.cam() = mr::math::Camera<float>(Vec3f{10, 10, 10}, Vec3f{-1, -1, -1}, Vec3f{0, 1, 0});
			if (edge(GLFW_KEY_3, k3))
				fps.cam() = mr::math::Camera<float>(Vec3f{100, 100, 100}, Vec3f{-1, -1, -1}, Vec3f{0, 1, 0});
			if (edge(GLFW_KEY_4, k4))
				fps.cam() = mr::math::Camera<float>(Vec3f{500, 500, 500}, Vec3f{-1, -1, -1}, Vec3f{0, 1, 0});
			if (edge(GLFW_KEY_5, k5))
				fps.cam() =
				    mr::math::Camera<float>(Vec3f{10000, 10000, 10000}, Vec3f{-1, -1, -1}, Vec3f{0, 1, 0});
			if (edge(GLFW_KEY_6, k6))
				fps.cam() = mr::math::Camera<float>(
				    Vec3f{100000, 100000, 100000}, Vec3f{-1, -1, -1}, Vec3f{0, 1, 0});
			if (edge(GLFW_KEY_0, k0)) {
				Vec3f cp = fps.cam().position();
				auto nd = cp.normalized();
				if (nd)
					fps.cam() = mr::math::Camera<float>(cp, -Vec3f(*nd), Vec3f{0, 1, 0});
			}
			if (edge(GLFW_KEY_ESCAPE, kEsc))
				glfwSetWindowShouldClose(win, GLFW_TRUE);

			Matr4f vp = fps.viewProj();
			runBenchmarkPass(vp, fps.cam().position(), BenchPrintStyle::LiveFour);
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
			for (size_t i = 0; i < objects.size(); ++i) {
				const SceneObject &o = objects[i];
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
			glfwSwapBuffers(win);
		}
		std::fprintf(stderr, "\n");
	}

	glDeleteRenderbuffers(1, &depthRb);
	glDeleteTextures(1, &colorTex);
	glDeleteFramebuffers(1, &fbo);
	glDeleteBuffers(1, &vbo);
	glDeleteBuffers(1, &ebo);
	glDeleteVertexArrays(1, &vao);
	glDeleteProgram(progPreview);
	glDeleteProgram(prog);
	glfwDestroyWindow(win);
	glfwTerminate();
	return 0;
}
