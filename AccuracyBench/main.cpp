////////////////////////////////////////////////////////////////////////////////
// AccuracyBench: compare MaskedOcclusionCulling vs GPU visibility (uint ID buffer).
////////////////////////////////////////////////////////////////////////////////

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include "../MaskedOcclusionCulling.h"

namespace {

constexpr int kFbW = 1280;
constexpr int kFbH = 720;

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

// Same MVP as ID pass; false-color RGB so you can see the scene in the window.
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

struct Mesh {
	std::vector<glm::vec3> positions;
	std::vector<unsigned int> indices;
	glm::vec3 aabbMin{0.f};
	glm::vec3 aabbMax{0.f};
};

struct SceneObject {
	Mesh mesh;
	glm::mat4 model{1.f};
	uint32_t id = 0;
};

static void ExpandAabb(glm::vec3 &mn, glm::vec3 &mx, const glm::vec3 &p) {
	mn = glm::min(mn, p);
	mx = glm::max(mx, p);
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
	Mesh m;
	const float v[8][3] = {
		{-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
		{-1, -1, 1}, {1, -1, 1}, {1, 1, 1}, {-1, 1, 1},
	};
	for (auto &c : v)
		m.positions.push_back(glm::vec3(c[0], c[1], c[2]) * 0.5f);
	const unsigned idx[] = {
		0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 5, 1, 0, 4, 5,
		2, 6, 7, 2, 7, 3, 0, 3, 7, 0, 7, 4, 1, 5, 6, 1, 6, 2,
	};
	for (unsigned i : idx)
		m.indices.push_back(i);
	MeshComputeAabb(m);
	return m;
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
		const auto &mesh = shape.mesh;
		if (mesh.indices.empty())
			continue;

		SceneObject obj;
		obj.id = nextId++;
		obj.model = glm::mat4(1.f);

		std::unordered_map<int, unsigned> remap;
		obj.mesh.indices.reserve(mesh.indices.size());
		for (size_t i = 0; i < mesh.indices.size(); ++i) {
			int vi = mesh.indices[i].vertex_index;
			if (vi < 0)
				continue;
			auto it = remap.find(vi);
			if (it == remap.end()) {
				unsigned ni = (unsigned)obj.mesh.positions.size();
				remap[vi] = ni;
				obj.mesh.positions.push_back(glm::vec3(
					attrib.vertices[3 * vi + 0],
					attrib.vertices[3 * vi + 1],
					attrib.vertices[3 * vi + 2]));
			}
			obj.mesh.indices.push_back(remap[vi]);
		}
		MeshComputeAabb(obj.mesh);
		if (!obj.mesh.indices.empty())
			out.push_back(std::move(obj));
	}
	return !out.empty();
}

static void MakeProceduralScene(std::vector<SceneObject> &out) {
	out.clear();
	unsigned n = 24;
	for (unsigned i = 0; i < n; ++i) {
		SceneObject o;
		o.id = i + 1;
		o.mesh = MakeUnitCube();
		float fx = float((i * 17) % 13) - 6.f;
		float fy = float((i * 7) % 9) - 3.f;
		float fz = -float(i % 10) * 2.2f - 2.f;
		o.model = glm::translate(glm::mat4(1.f), glm::vec3(fx, fy, fz));
		o.model = glm::scale(o.model, glm::vec3(0.22f + 0.015f * float(i % 5)));
		out.push_back(std::move(o));
	}
}

static void ObjectWorldCorners(const SceneObject &o, glm::vec3 wOut[8]) {
	const glm::vec3 &mn = o.mesh.aabbMin;
	const glm::vec3 &mx = o.mesh.aabbMax;
	const glm::vec3 lc[8] = {
		{mn.x, mn.y, mn.z}, {mx.x, mn.y, mn.z},
		{mx.x, mx.y, mn.z}, {mn.x, mx.y, mn.z},
		{mn.x, mn.y, mx.z}, {mx.x, mn.y, mx.z},
		{mx.x, mx.y, mx.z}, {mn.x, mx.y, mx.z},
	};
	for (int i = 0; i < 8; ++i)
		wOut[i] = glm::vec3(o.model * glm::vec4(lc[i], 1.f));
}

static void FrustumPlanesFromVp(const glm::mat4 &vp, glm::vec4 planes[6]) {
	glm::mat4 t = glm::transpose(vp);
	glm::vec4 r0 = t[0], r1 = t[1], r2 = t[2], r3 = t[3];
	planes[0] = glm::normalize(r3 + r0);
	planes[1] = glm::normalize(r3 - r0);
	planes[2] = glm::normalize(r3 + r1);
	planes[3] = glm::normalize(r3 - r1);
	planes[4] = glm::normalize(r3 + r2);
	planes[5] = glm::normalize(r3 - r2);
}

// Homogeneous clip-space frustum vs world AABB (8 corners → clip).
static bool ObjectIntersectsFrustum(const glm::mat4 &vp, const glm::vec3 wCorners[8]) {
	glm::vec4 planes[6];
	FrustumPlanesFromVp(vp, planes);
	for (int p = 0; p < 6; ++p) {
		int outside = 0;
		for (int i = 0; i < 8; ++i) {
			glm::vec4 cl = vp * glm::vec4(wCorners[i], 1.f);
			if (glm::dot(planes[p], cl) < 0.f)
				outside++;
		}
		if (outside == 8)
			return false;
	}
	return true;
}

static float DepthKey(const glm::vec3 &worldCenter, const glm::vec3 &camPos) {
	return glm::length(worldCenter - camPos);
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

// Camera: position, view direction, up (world). lookAt center = position + direction.
// Formats:
//   ((px,py,pz),(dx,dy,dz),(ux,uy,uz))   — quote in shell: bash treats ((…)) specially without quotes
//   px,py,pz|dx,dy,dz|ux,uy,uz           — safe without parentheses
static bool ParseCameraSpec(const char *s, glm::vec3 &outPos, glm::vec3 &outDir, glm::vec3 &outUp) {
	float px, py, pz, dx, dy, dz, ux, uy, uz;
	int n = std::sscanf(s, "%f,%f,%f|%f,%f,%f|%f,%f,%f",
	    &px, &py, &pz, &dx, &dy, &dz, &ux, &uy, &uz);
	if (n != 9)
		n = std::sscanf(s, "((%f,%f,%f),(%f,%f,%f),(%f,%f,%f))",
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
	outPos = glm::vec3(px, py, pz);
	outDir = glm::vec3(dx, dy, dz);
	outUp = glm::vec3(ux, uy, uz);
	if (glm::length(outDir) < 1e-9f) {
		std::fprintf(stderr, "Camera: direction vector length is ~0.\n");
		return false;
	}
	if (glm::length(outUp) < 1e-9f) {
		std::fprintf(stderr, "Camera: up vector length is ~0.\n");
		return false;
	}
	return true;
}

} // namespace

int main(int argc, char **argv) {
	bool headless = false;
	const char *objPath = nullptr;
	const char *cameraSpec = nullptr;
	for (int i = 1; i < argc; ++i) {
		if (!std::strcmp(argv[i], "--headless")) {
			headless = true;
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
		if (argv[i][0] != '-' && !objPath)
			objPath = argv[i];
	}

	std::vector<SceneObject> objects;
	if (objPath) {
		if (!LoadObjMeshes(objPath, objects)) {
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

	if (!glfwInit()) {
		fprintf(stderr, "glfwInit failed\n");
		return 1;
	}
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_VISIBLE, headless ? GLFW_FALSE : GLFW_TRUE);
	GLFWwindow *win = glfwCreateWindow(kFbW, kFbH, "AccuracyBench - preview (close to exit)", nullptr, nullptr);
	if (!win) {
		fprintf(stderr, "glfwCreateWindow failed\n");
		return 1;
	}
	glfwMakeContextCurrent(win);
	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "glewInit failed\n");
		return 1;
	}
	glGetError(); // clear GLEW noise

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
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32UI, kFbW, kFbH, 0, GL_RGBA_INTEGER, GL_UNSIGNED_INT, nullptr);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorTex, 0);
	glGenRenderbuffers(1, &depthRb);
	glBindRenderbuffer(GL_RENDERBUFFER, depthRb);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, kFbW, kFbH);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthRb);
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
		fprintf(stderr, "FBO incomplete\n");
		return 1;
	}
	CheckGl("FBO setup");

	const float nearP = 1.0f;
	const float farP = 250.f;
	const float aspect = float(kFbW) / float(kFbH);
	glm::mat4 proj = glm::perspective(glm::radians(55.f), aspect, nearP, farP);

	glm::vec3 camPos, camDir, camUp;
	if (cameraSpec) {
		std::fprintf(stderr, "Parsing --camera: %s\n", cameraSpec);
		if (!ParseCameraSpec(cameraSpec, camPos, camDir, camUp))
			return 1;
	} else {
		camPos = glm::vec3(0.f, 5.f, 42.f);
		camDir = glm::vec3(0.f, 0.f, 0.f) - camPos;
		camUp = glm::vec3(0.f, 1.f, 0.f);
	}
	glm::vec3 target = camPos + camDir;
	glm::mat4 view = glm::lookAt(camPos, target, camUp);
	glm::mat4 vp = proj * view;

	MaskedOcclusionCulling *moc = MaskedOcclusionCulling::Create();
	moc->SetResolution((unsigned)kFbW, (unsigned)kFbH);
	moc->SetNearClipPlane(nearP);
	moc->ClearBuffer();

	std::vector<size_t> order(objects.size());
	for (size_t i = 0; i < order.size(); ++i)
		order[i] = i;
	std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
		glm::vec3 ca = (objects[a].model * glm::vec4(
			(objects[a].mesh.aabbMin + objects[a].mesh.aabbMax) * 0.5f, 1.f));
		glm::vec3 cb = (objects[b].model * glm::vec4(
			(objects[b].mesh.aabbMin + objects[b].mesh.aabbMax) * 0.5f, 1.f));
		return DepthKey(glm::vec3(ca), camPos) < DepthKey(glm::vec3(cb), camPos);
	});

	for (size_t k : order) {
		const SceneObject &o = objects[k];
		glm::mat4 mvp = vp * o.model;
		const Mesh &mesh = o.mesh;
		std::vector<float> clipVerts(mesh.positions.size() * 4);
		MaskedOcclusionCulling::TransformVertices(
			glm::value_ptr(mvp),
			&mesh.positions[0].x,
			&clipVerts[0],
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

	for (size_t i = 0; i < objects.size(); ++i) {
		const SceneObject &o = objects[i];
		glm::vec3 wcorners[8];
		ObjectWorldCorners(o, wcorners);
		bool inf = ObjectIntersectsFrustum(vp, wcorners);
		frustumHit[i] = inf ? 1 : 0;
		if (inf)
			++nFrustum;
	}

	for (size_t i = 0; i < objects.size(); ++i) {
		if (!frustumHit[i])
			continue;
		const SceneObject &o = objects[i];
		glm::mat4 mvp = vp * o.model;
		const Mesh &mesh = o.mesh;
		std::vector<float> clipVerts(mesh.positions.size() * 4);
		MaskedOcclusionCulling::TransformVertices(
			glm::value_ptr(mvp),
			&mesh.positions[0].x,
			&clipVerts[0],
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

	glViewport(0, 0, kFbW, kFbH);
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
		glm::mat4 mvp = vp * o.model;
		glUniformMatrix4fv(locMvp, 1, GL_FALSE, glm::value_ptr(mvp));
		glUniform1ui(locId, o.id);

		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, o.mesh.positions.size() * sizeof(glm::vec3),
			o.mesh.positions.data(), GL_STREAM_DRAW);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, o.mesh.indices.size() * sizeof(unsigned),
			o.mesh.indices.data(), GL_STREAM_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void *)0);

		glDrawElements(GL_TRIANGLES, (GLsizei)o.mesh.indices.size(), GL_UNSIGNED_INT, nullptr);
	}
	CheckGl("draw");
	glFinish();

	std::vector<unsigned> pixels((size_t)kFbW * (size_t)kFbH * 4);
	glReadBuffer(GL_COLOR_ATTACHMENT0);
	glReadPixels(0, 0, kFbW, kFbH, GL_RGBA_INTEGER, GL_UNSIGNED_INT, pixels.data());
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

	printf("\n");
	printf("=== How geometry is drawn (same MVP on CPU and GPU) ===\n");
	printf(
 "  Mesh vertices are in object/local space (OBJ shape or unit cube).\n"
	    "  MVP = projection * view * model (glm, column-major).\n"
	    "  MOC: TransformVertices(MVP, local xyz) -> clip (x,y,z,w), then RenderTriangles /\n"
	    "       TestTriangles on that clip stream (see MaskedOcclusionCulling README: uses w, ~1/w depth).\n"
	    "  OpenGL: VBO = same local positions; vertex shader gl_Position = MVP * vec4(aPos,1).\n"
	    "          Fragment shader writes uvec4(0,0,0, objectId) to RGBA32UI; depth buffer GL_LESS.\n"
	    "  Frustum column: AABB corners transformed model->world, then VP->clip; reject if all 8\n"
	    "          corners lie outside one frustum plane (homogeneous clip test).\n\n");
	printf("Camera pos (%.2f, %.2f, %.2f)  dir (%.2f, %.2f, %.2f)  up (%.2f, %.2f, %.2f)\n",
	    camPos.x, camPos.y, camPos.z, camDir.x, camDir.y, camDir.z, camUp.x, camUp.y, camUp.z);
	printf("  -> lookAt target (%.2f, %.2f, %.2f)   FOV 55 deg   near %.2f  far %.2f\n",
	    target.x, target.y, target.z, nearP, farP);

	printf("\n=== Per-object (translation = model matrix column 3; id = GPU/MRT .a) ===\n");
	printf(
	    "%-4s %-5s %5s %5s %10s %10s %10s  %-7s  %-14s  %s\n",
	    "#", "id", "vtx", "tri", "tx", "ty", "tz", "frustum", "MOC", "GPU_pixel");
	for (size_t i = 0; i < objects.size(); ++i) {
		const SceneObject &o = objects[i];
		glm::vec3 t = glm::vec3(o.model[3]);
		const char *fr = frustumHit[i] ? "yes" : "no";
		const char *mocS = mocTested[i] ? MocResultStr(mocRaw[i]) : "—";
		const char *gpuS = gpuVis[i] ? "yes" : "no";
		printf(
		    "%-4zu %-5u %5zu %5zu %10.2f %10.2f %10.2f  %-7s  %-14s  %s\n",
		    i, o.id, o.mesh.positions.size(), o.mesh.indices.size() / 3, t.x, t.y, t.z, fr, mocS, gpuS);
	}

	printf("\n--- AccuracyBench (MaskedOcclusionCulling vs RGBA32UI id in .a) ---\n");
	printf("Resolution %dx%d  MOC USE_D3D=%d (see MaskedOcclusionCulling.h)  near=%.2f\n",
		kFbW, kFbH, USE_D3D, nearP);
	printf("1) All objects:              %u\n", nAll);
	printf("2) AABB in frustum:          %u\n", nFrustum);
	printf("3) MOC TestTriangles VISIBLE (subset of frustum): %u\n", nMocVisible);
	printf("4) GPU unique object IDs in buffer (any pixel):   %u\n", nGpuVisible);
	printf("--- errors (frustum subset only) ---\n");
	printf("False positives (MOC visible, GPU no pixel): %u\n", fp);
	printf("False negatives (MOC occluded/culled, GPU pixel): %u\n", fn);
	printf("(Conservative culling should avoid false negatives; a non-zero count means mismatch.)\n");
	if (!headless) {
		printf("\nOpening preview window: each object is a distinct false color (same id as G-buffer .a).\n");
		printf("Close the window to quit.\n");
	}
	fflush(stdout);

	if (!headless) {
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		int winW = kFbW, winH = kFbH;
		glfwGetFramebufferSize(win, &winW, &winH);
		glViewport(0, 0, winW, winH);
		glClearColor(0.02f, 0.02f, 0.03f, 1.f);
		glClear(GL_COLOR_BUFFER_BIT);
		// Same aspect and vp as offscreen FBO so the preview matches the benchmark view
		// (letterbox on HiDPI / resized windows instead of stretching projection).
		const float scale = std::min(winW / float(kFbW), winH / float(kFbH));
		const int vw = std::max(1, (int)(kFbW * scale + 0.5f));
		const int vh = std::max(1, (int)(kFbH * scale + 0.5f));
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
			glm::mat4 mvp = vp * o.model;
			glUniformMatrix4fv(locMvpPrev, 1, GL_FALSE, glm::value_ptr(mvp));
			glUniform1ui(locIdPrev, o.id);
			glBindBuffer(GL_ARRAY_BUFFER, vbo);
			glBufferData(GL_ARRAY_BUFFER, o.mesh.positions.size() * sizeof(glm::vec3),
				o.mesh.positions.data(), GL_STREAM_DRAW);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, o.mesh.indices.size() * sizeof(unsigned),
				o.mesh.indices.data(), GL_STREAM_DRAW);
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void *)0);
			glDrawElements(GL_TRIANGLES, (GLsizei)o.mesh.indices.size(), GL_UNSIGNED_INT, nullptr);
		}
		CheckGl("preview draw");
		glfwSwapBuffers(win);

		while (!glfwWindowShouldClose(win)) {
			glfwPollEvents();
		}
	}

	glDeleteRenderbuffers(1, &depthRb);
	glDeleteTextures(1, &colorTex);
	glDeleteFramebuffers(1, &fbo);
	glDeleteBuffers(1, &vbo);
	glDeleteBuffers(1, &ebo);
	glDeleteVertexArrays(1, &vao);
	glDeleteProgram(progPreview);
	glDeleteProgram(prog);
	MaskedOcclusionCulling::Destroy(moc);
	glfwDestroyWindow(win);
	glfwTerminate();
	return 0;
}
