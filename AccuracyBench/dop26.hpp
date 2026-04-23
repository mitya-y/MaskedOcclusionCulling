#ifndef ACCBENCH_DOP26_HPP
#define ACCBENCH_DOP26_HPP

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

/// 26-dop (Bartz et al.): 13 fixed directions, min/max (dot) per direction.
/// Vertices and edges in local space; main.cpp projects to NDC (same w-clip as AABB).
namespace accbench::dop26 {

struct Dop3f {
	float x, y, z;
};

void BuildDop26FromVertexPositions(
    const float* positionsXyz,
    std::size_t numPoints,
    std::size_t strideBytes,
    std::vector<Dop3f> &outVerts,
    std::vector<std::pair<std::uint16_t, std::uint16_t>> &outEdges);

} // namespace accbench::dop26

#endif
