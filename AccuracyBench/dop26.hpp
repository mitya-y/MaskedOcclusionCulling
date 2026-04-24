#ifndef ACCBENCH_DOP26_HPP
#define ACCBENCH_DOP26_HPP

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

/// K-DOP in local space: k/2 fixed axis directions, min/max (dot) per direction.
/// Supported k: even, 10…30 (see IsSupportedDopK). k=26 uses Bartz 13-dir set;
/// m<13 take first m of same ordered set; m=14,15 append two extra integer directions.
/// Vertices and edges in local space; main.cpp projects to NDC (same w-clip as AABB).
namespace accbench::dop26 {

struct Dop3f {
	float x, y, z;
};

/// True if k is even, 10 <= k <= 30 (K-DOP in this build).
bool IsSupportedDopK(int k);

void BuildDopKFromVertexPositions(
    const float *positionsXyz,
    std::size_t numPoints,
    std::size_t strideBytes,
    int k,
    std::vector<Dop3f> &outVerts,
    std::vector<std::pair<std::uint16_t, std::uint16_t>> &outEdges);

void BuildDop26FromVertexPositions(
    const float *positionsXyz,
    std::size_t numPoints,
    std::size_t strideBytes,
    std::vector<Dop3f> &outVerts,
    std::vector<std::pair<std::uint16_t, std::uint16_t>> &outEdges);

} // namespace accbench::dop26

#endif
