#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace accbench::usd {

struct MeshBuffers {
	std::vector<std::array<float, 3>> positions;
	std::vector<uint32_t> indices;
};

struct Instance {
	// Column-major 4×4 for OpenGL / Matr4FromColumnMajor in main.cpp
	float modelColumnMajor[16]{};
	int meshIndex = -1;
};

// Loads USDA / USDC / USDZ via TinyUSDZ + Tydra (triangulated meshes).
bool LoadStage(const char *path, std::vector<MeshBuffers> &meshes, std::vector<Instance> &instances,
    std::string &warn, std::string &err);

} // namespace accbench::usd
