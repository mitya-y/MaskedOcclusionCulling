#include "dop26.hpp"

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cstdint>
#include <set>
#include <utility>
#include <vector>

namespace accbench::dop26 {
namespace {

// Bartz et al. (Tighter Bounding Volumes for Better Occlusion Cull), Sec. 3.3
static const int kNDirs = 13;
static const float kDir[kNDirs][3] = {
    {1, 0, 0},
    {0, 1, 0},
    {0, 0, 1},
    {1, 1, 0},
    {1, 0, 1},
    {0, 1, 1},
    {1, -1, 0},
    {1, 0, -1},
    {0, 1, -1},
    {1, 1, 1},
    {1, -1, 1},
    {1, 1, -1},
    {-1, 1, 1},
};

struct Hp {
	float ax, ay, az, b; // ax*x+ay*y+az*z >= b
};

struct Vec3 {
	float x, y, z;
};

static inline float Dot(const Hp &h, float x, float y, float z) {
	return h.ax * x + h.ay * y + h.az * z;
}

static inline Vec3 Cross(const float u[3], const float v[3]) {
	return Vec3{
	    u[1] * v[2] - u[2] * v[1],
	    u[2] * v[0] - u[0] * v[2],
	    u[0] * v[1] - u[1] * v[0],
	};
}

static void MeshExtent(
    const float* positionsXyz, std::size_t numPoints, std::size_t stride, float& outR) {
	if (numPoints == 0) {
		outR = 1.f;
		return;
	}
	const unsigned char* base = reinterpret_cast<const unsigned char*>(positionsXyz);
	float mx = 0.f, my = 0.f, mz = 0.f, Mx = 0.f, My = 0.f, Mz = 0.f;
	for (std::size_t i = 0; i < numPoints; ++i) {
		const float* p = reinterpret_cast<const float*>(base + i * stride);
		if (i == 0) {
			mx = Mx = p[0];
			my = My = p[1];
			mz = Mz = p[2];
		} else {
			mx = std::min(mx, p[0]);
			my = std::min(my, p[1]);
			mz = std::min(mz, p[2]);
			Mx = std::max(Mx, p[0]);
			My = std::max(My, p[1]);
			Mz = std::max(Mz, p[2]);
		}
	}
	float ex = Mx - mx, ey = My - my, ez = Mz - mz;
	float e = std::max(ex, std::max(ey, ez));
	outR = (e > 1e-20f) ? e : 1.f;
}

static bool InsideSlack(const Hp* hp, float x, float y, float z, float negSlack) {
	for (int i = 0; i < 26; ++i) {
		if (Dot(hp[i], x, y, z) < hp[i].b + negSlack)
			return false;
	}
	return true;
}

// Rows: a1·(x,y,z) = b1, a2·= b2, a3·= b3
static bool solve3x3(
    const float a1[3], float b1, const float a2[3], float b2, const float a3[3], float b3, float& x,
    float& y, float& z) {
	const float* r1 = a1, *r2 = a2, *r3 = a3;
	const float c1 = b1, c2 = b2, c3 = b3;
	float detA = r1[0] * (r2[1] * r3[2] - r2[2] * r3[1]) - r1[1] * (r2[0] * r3[2] - r2[2] * r3[0]) +
	    r1[2] * (r2[0] * r3[1] - r2[1] * r3[0]);
	if (std::fabs(detA) < 1e-24f)
		return false;
	// Cramer: column 0 <- (c1,c2,c3)
	float detX = c1 * (r2[1] * r3[2] - r2[2] * r3[1]) - r1[1] * (c2 * r3[2] - r2[2] * c3) + r1[2] * (c2 * r3[1] - r2[1] * c3);
	float detY = r1[0] * (c2 * r3[2] - r2[2] * c3) - c1 * (r2[0] * r3[2] - r2[2] * r3[0]) + r1[2] * (r2[0] * c3 - c2 * r3[0]);
	float detZ = r1[0] * (r2[1] * c3 - c2 * r3[1]) - r1[1] * (r2[0] * c3 - c2 * r3[0]) + c1 * (r2[0] * r3[1] - r2[1] * r3[0]);
	x = detX / detA;
	y = detY / detA;
	z = detZ / detA;
	return true;
}

static bool pickP0(
    const float a[3], float b, const float c[3], float d, const Vec3& u, float& px, float& py, float& pz) {
	// a·p = b, c·p = d, and u is line direction: pick p0 on that line
	// u·(p0 + t u) = u·p0 + t|u|²; try 2x2 + 1 coordinate zero
	(void)u;
	// z = 0
	{
		float det = a[0] * c[1] - a[1] * c[0];
		if (std::fabs(det) > 1e-20f) {
			px = (b * c[1] - a[1] * d) / det;
			py = (a[0] * d - b * c[0]) / det;
			pz = 0.f;
			return true;
		}
	}
	// y = 0
	{
		float det = a[0] * c[2] - a[2] * c[0];
		if (std::fabs(det) > 1e-20f) {
			px = (b * c[2] - a[2] * d) / det;
			py = 0.f;
			pz = (a[0] * d - b * c[0]) / det;
			return true;
		}
	}
	// x = 0
	{
		float det = a[1] * c[2] - a[2] * c[1];
		if (std::fabs(det) > 1e-20f) {
			px = 0.f;
			py = (b * c[2] - a[2] * d) / det;
			pz = (a[1] * d - b * c[1]) / det;
			return true;
		}
	}
	return false;
}

static void clipTInterval(
    const Hp& hp, float p0x, float p0y, float p0z, float ux, float uy, float uz, float& t0, float& t1) {
	const float A = hp.ax * ux + hp.ay * uy + hp.az * uz;
	const float B = hp.ax * p0x + hp.ay * p0y + hp.az * p0z;
	if (std::fabs(A) < 1e-20f) {
		if (B < hp.b - 1e-6f) {
			t0 = 1.f;
			t1 = 0.f; // infeasible
		}
		return;
	}
	const float tBound = (hp.b - B) / A;
	if (A > 0.f) {
		if (tBound > t0)
			t0 = tBound;
	} else {
		if (tBound < t1)
			t1 = tBound;
	}
}

} // namespace

void BuildDop26FromVertexPositions(
    const float* positionsXyz, std::size_t numPoints, std::size_t strideBytes, std::vector<Dop3f> &outVerts,
    std::vector<std::pair<std::uint16_t, std::uint16_t>> &outEdges) {
	outVerts.clear();
	outEdges.clear();
	if (!positionsXyz || numPoints < 1)
		return;

	float meshR = 1.f;
	MeshExtent(positionsXyz, numPoints, strideBytes, meshR);
	const float inEps = -1e-4f * meshR;

	// 13 min/max
	float dmin[kNDirs], dmax[kNDirs];
	for (int d = 0; d < kNDirs; ++d) {
		dmin[d] = FLT_MAX;
		dmax[d] = -FLT_MAX;
	}
	const unsigned char* base = reinterpret_cast<const unsigned char*>(positionsXyz);
	for (std::size_t i = 0; i < numPoints; ++i) {
		const float* p = reinterpret_cast<const float*>(base + i * strideBytes);
		for (int d = 0; d < kNDirs; ++d) {
			float t = kDir[d][0] * p[0] + kDir[d][1] * p[1] + kDir[d][2] * p[2];
			dmin[d] = std::min(dmin[d], t);
			dmax[d] = std::max(dmax[d], t);
		}
	}

	// 26 halfspaces: d·p >= min, (-d)·p >= -max
	Hp hp[26];
	for (int d = 0; d < kNDirs; ++d) {
		hp[2 * d + 0] = Hp{kDir[d][0], kDir[d][1], kDir[d][2], dmin[d]};
		hp[2 * d + 1] = Hp{-kDir[d][0], -kDir[d][1], -kDir[d][2], -dmax[d]};
	}

	// Feasible region non-empty: check one vertex (center of AABB) roughly - skip

	std::vector<float> candX, candY, candZ;
	candX.reserve(128);
	candY.reserve(128);
	candZ.reserve(128);

	for (int i = 0; i < 26; ++i) {
		for (int j = i + 1; j < 26; ++j) {
			for (int k = j + 1; k < 26; ++k) {
				const float r1[3] = {hp[i].ax, hp[i].ay, hp[i].az};
				const float r2[3] = {hp[j].ax, hp[j].ay, hp[j].az};
				const float r3[3] = {hp[k].ax, hp[k].ay, hp[k].az};
				float x, y, z;
				if (!solve3x3(r1, hp[i].b, r2, hp[j].b, r3, hp[k].b, x, y, z))
					continue;
				if (InsideSlack(hp, x, y, z, inEps)) {
					candX.push_back(x);
					candY.push_back(y);
					candZ.push_back(z);
				}
			}
		}
	}

	const float mergeEps = 1e-4f * meshR;
	auto snapVert = [&](float x, float y, float z) -> std::uint16_t {
		for (std::size_t n = 0; n < outVerts.size(); ++n) {
			const Dop3f& v = outVerts[n];
			float dx = v.x - x, dy = v.y - y, dz = v.z - z;
			if (dx * dx + dy * dy + dz * dz <= mergeEps * mergeEps)
				return static_cast<std::uint16_t>(n);
		}
		if (outVerts.size() >= 65535) {
			return 0; // should not happen for dops
		}
		outVerts.push_back(Dop3f{x, y, z});
		return static_cast<std::uint16_t>(outVerts.size() - 1);
	};

	// Deduplicate candidates into outVerts
	for (std::size_t t = 0; t < candX.size(); ++t)
		(void)snapVert(candX[t], candY[t], candZ[t]);

	if (outVerts.empty())
		return;

	std::set<std::pair<std::uint16_t, std::uint16_t>> edgeSeen;

	// Edges: clip line from intersection of 2 face equalities
	for (int e1 = 0; e1 < 26; ++e1) {
		const float a1[3] = {hp[e1].ax, hp[e1].ay, hp[e1].az};
		for (int e2 = e1 + 1; e2 < 26; ++e2) {
			const float a2[3] = {hp[e2].ax, hp[e2].ay, hp[e2].az};
			Vec3 u = Cross(a1, a2);
			float ulen2 = u.x * u.x + u.y * u.y + u.z * u.z;
			if (ulen2 < 1e-30f)
				continue;
			float p0x, p0y, p0z;
			if (!pickP0(a1, hp[e1].b, a2, hp[e2].b, u, p0x, p0y, p0z))
				continue;
			float tLo = -1e20f, tHi = 1e20f;
			for (int h = 0; h < 26; ++h)
				clipTInterval(hp[h], p0x, p0y, p0z, u.x, u.y, u.z, tLo, tHi);
			if (tLo > tHi - 1e-6f * meshR)
				continue;
			// Shrink slightly to avoid spurious out-of-slab
			float tA = tLo;
			float tB = tHi;
			float sx = p0x + tA * u.x, sy = p0y + tA * u.y, sz = p0z + tA * u.z;
			float ex = p0x + tB * u.x, ey = p0y + tB * u.y, ez = p0z + tB * u.z;
			if (!InsideSlack(hp, sx, sy, sz, inEps) || !InsideSlack(hp, ex, ey, ez, inEps)) {
				// line segment might be invalid due to num — skip
				continue;
			}
			std::uint16_t ia = snapVert(sx, sy, sz);
			std::uint16_t ib = snapVert(ex, ey, ez);
			if (ia == ib)
				continue;
			std::uint16_t lo = (ia < ib) ? ia : ib;
			std::uint16_t hi = (ia < ib) ? ib : ia;
			edgeSeen.emplace(lo, hi);
		}
	}

	for (const auto& e : edgeSeen)
		outEdges.push_back({e.first, e.second});
}

} // namespace accbench::dop26
