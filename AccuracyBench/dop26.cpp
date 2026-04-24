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

// Ordered direction table: Bartz 13 (Sec. 3.3) + 2 for k=28/30 (m=14,15). First m used for k=2m.
static const int kMaxM = 15;
static const float kAllDirs[kMaxM][3] = {
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
    {1, 2, 0},
    {1, 0, 2},
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
    const float *positionsXyz, std::size_t numPoints, std::size_t stride, float &outR) {
	if (numPoints == 0) {
		outR = 1.f;
		return;
	}
	const unsigned char *base = reinterpret_cast<const unsigned char *>(positionsXyz);
	float mx = 0.f, my = 0.f, mz = 0.f, Mx = 0.f, My = 0.f, Mz = 0.f;
	for (std::size_t i = 0; i < numPoints; ++i) {
		const float *p = reinterpret_cast<const float *>(base + i * stride);
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

static bool InsideSlack(const Hp *hp, int nHp, float x, float y, float z, float negSlack) {
	for (int i = 0; i < nHp; ++i) {
		if (Dot(hp[i], x, y, z) < hp[i].b + negSlack)
			return false;
	}
	return true;
}

// Rows: a1·(x,y,z) = b1, a2·= b2, a3·= b3
static bool solve3x3(
    const float a1[3], float b1, const float a2[3], float b2, const float a3[3], float b3, float &x,
    float &y, float &z) {
	const float *r1 = a1, *r2 = a2, *r3 = a3;
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
    const float a[3], float b, const float c[3], float d, const Vec3 &u, float &px, float &py, float &pz) {
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
    const Hp &hp, float p0x, float p0y, float p0z, float ux, float uy, float uz, float &t0, float &t1) {
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

bool IsSupportedDopK(int k) {
	if ((k & 1) != 0)
		return false;
	return k >= 10 && k <= 30;
}

void BuildDopKFromVertexPositions(
    const float *positionsXyz, std::size_t numPoints, std::size_t strideBytes, int k, std::vector<Dop3f> &outVerts,
    std::vector<std::pair<std::uint16_t, std::uint16_t>> &outEdges) {
	outVerts.clear();
	outEdges.clear();
	if (!IsSupportedDopK(k) || !positionsXyz || numPoints < 1)
		return;
	const int m = k / 2;
	if (m < 1 || m > kMaxM)
		return;

	float meshR = 1.f;
	MeshExtent(positionsXyz, numPoints, strideBytes, meshR);
	const float inEps = -1e-4f * meshR;

	float dmin[kMaxM], dmax[kMaxM];
	for (int d = 0; d < m; ++d) {
		dmin[d] = FLT_MAX;
		dmax[d] = -FLT_MAX;
	}
	const unsigned char *base = reinterpret_cast<const unsigned char *>(positionsXyz);
	for (std::size_t i = 0; i < numPoints; ++i) {
		const float *p = reinterpret_cast<const float *>(base + i * strideBytes);
		for (int d = 0; d < m; ++d) {
			float t = kAllDirs[d][0] * p[0] + kAllDirs[d][1] * p[1] + kAllDirs[d][2] * p[2];
			dmin[d] = std::min(dmin[d], t);
			dmax[d] = std::max(dmax[d], t);
		}
	}

	const int nHp = 2 * m;
	std::vector<Hp> hp(static_cast<std::size_t>(nHp));
	for (int d = 0; d < m; ++d) {
		hp[2 * d + 0] = Hp{kAllDirs[d][0], kAllDirs[d][1], kAllDirs[d][2], dmin[d]};
		hp[2 * d + 1] = Hp{-kAllDirs[d][0], -kAllDirs[d][1], -kAllDirs[d][2], -dmax[d]};
	}
	const Hp *hpData = hp.data();

	std::vector<float> candX, candY, candZ;
	candX.reserve(256);
	candY.reserve(256);
	candZ.reserve(256);

	for (int i = 0; i < nHp; ++i) {
		for (int j = i + 1; j < nHp; ++j) {
			for (int kk = j + 1; kk < nHp; ++kk) {
				const float r1[3] = {hpData[i].ax, hpData[i].ay, hpData[i].az};
				const float r2[3] = {hpData[j].ax, hpData[j].ay, hpData[j].az};
				const float r3[3] = {hpData[kk].ax, hpData[kk].ay, hpData[kk].az};
				float x, y, z;
				if (!solve3x3(r1, hpData[i].b, r2, hpData[j].b, r3, hpData[kk].b, x, y, z))
					continue;
				if (InsideSlack(hpData, nHp, x, y, z, inEps)) {
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
			const Dop3f &v = outVerts[n];
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

	for (std::size_t t = 0; t < candX.size(); ++t)
		(void)snapVert(candX[t], candY[t], candZ[t]);

	if (outVerts.empty())
		return;

	std::set<std::pair<std::uint16_t, std::uint16_t>> edgeSeen;

	for (int e1 = 0; e1 < nHp; ++e1) {
		const float a1[3] = {hpData[e1].ax, hpData[e1].ay, hpData[e1].az};
		for (int e2 = e1 + 1; e2 < nHp; ++e2) {
			const float a2[3] = {hpData[e2].ax, hpData[e2].ay, hpData[e2].az};
			Vec3 u = Cross(a1, a2);
			float ulen2 = u.x * u.x + u.y * u.y + u.z * u.z;
			if (ulen2 < 1e-30f)
				continue;
			float p0x, p0y, p0z;
			if (!pickP0(a1, hpData[e1].b, a2, hpData[e2].b, u, p0x, p0y, p0z))
				continue;
			float tLo = -1e20f, tHi = 1e20f;
			for (int h = 0; h < nHp; ++h)
				clipTInterval(hpData[h], p0x, p0y, p0z, u.x, u.y, u.z, tLo, tHi);
			if (tLo > tHi - 1e-6f * meshR)
				continue;
			float tA = tLo;
			float tB = tHi;
			float sx = p0x + tA * u.x, sy = p0y + tA * u.y, sz = p0z + tA * u.z;
			float ex = p0x + tB * u.x, ey = p0y + tB * u.y, ez = p0z + tB * u.z;
			if (!InsideSlack(hpData, nHp, sx, sy, sz, inEps) || !InsideSlack(hpData, nHp, ex, ey, ez, inEps)) {
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

	for (const auto &e : edgeSeen)
		outEdges.push_back({e.first, e.second});
}

void BuildDop26FromVertexPositions(
    const float *positionsXyz, std::size_t numPoints, std::size_t strideBytes, std::vector<Dop3f> &outVerts,
    std::vector<std::pair<std::uint16_t, std::uint16_t>> &outEdges) {
	BuildDopKFromVertexPositions(positionsXyz, numPoints, strideBytes, 26, outVerts, outEdges);
}

} // namespace accbench::dop26
