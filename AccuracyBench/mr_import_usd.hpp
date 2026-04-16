#pragma once

#include "usd_load.hpp"

#include <string>

namespace accbench::mrimp {

// Uses 4J mr-importer (OpenUSD, glTF, … via mr::importer::import). Requires
// ACCURACYBENCH_USE_MR_IMPORTER=ON and a Conan/toolchain that provides mr-importer (see CMakeLists.txt).
bool LoadStage(const char *path, std::vector<accbench::usd::MeshBuffers> &meshes,
    std::vector<accbench::usd::Instance> &instances, std::string &err);

} // namespace accbench::mrimp
