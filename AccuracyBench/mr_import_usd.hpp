#pragma once

#include "usd_load.hpp"

#include <string>

namespace accbench::mrimp {

// Uses 4J mr-importer (OpenUSD). Requires ACCURACYBENCH_USE_MR_IMPORTER=ON and
// a Conan/toolchain that provides mr-importer + pxr (see AccuracyBench/CMakeLists.txt).
bool LoadStage(const char *path, std::vector<accbench::usd::MeshBuffers> &meshes,
    std::vector<accbench::usd::Instance> &instances, std::string &err);

} // namespace accbench::mrimp
