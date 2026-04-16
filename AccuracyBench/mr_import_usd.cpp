#include "mr_import_usd.hpp"

#include <mr-importer/importer.hpp>

#include <mr-math/math.hpp>

#include <cstdio>
#include <filesystem>

namespace accbench::mrimp {
namespace {

static void Matr4fToColumnMajorGl(const mr::Matr4f &m, float out[16]) {
	for (int row = 0; row < 4; ++row)
		for (int col = 0; col < 4; ++col)
			out[col * 4 + row] = m[static_cast<size_t>(col)][static_cast<size_t>(row)];
}

} // namespace

bool LoadStage(const char *path, std::vector<accbench::usd::MeshBuffers> &meshes,
    std::vector<accbench::usd::Instance> &instances, std::string &err) {
	err.clear();
	meshes.clear();
	instances.clear();
	if (!path || !path[0]) {
		err = "empty path";
		return false;
	}

	std::fprintf(stderr, "mr-importer: mr::import starting…\n");
	std::fflush(stderr);

	using enum mr::importer::Options;
	mr::importer::Options opts = mr::importer::Options::All;
	opts = mr::importer::Options(opts & ~mr::importer::Options::PreferUncompressed);
	opts = mr::importer::Options(
	    opts & ~(mr::importer::Options::GenerateDiscreteLODs | mr::importer::Options::OptimizeMeshes |
			mr::importer::Options::GenerateMeshlets));
	opts = mr::importer::Options(opts & ~mr::importer::Options::LoadMaterials);
	opts = mr::importer::Options(opts & ~mr::importer::Options::LoadMeshAttributes);

	auto model = mr::importer::import(std::filesystem::path(path), opts);
	if (!model) {
		err = "mr::import returned nullopt";
		return false;
	}

	const auto &mv = *model;
	meshes.reserve(mv.meshes.size());
	for (const auto &mm : mv.meshes) {
		accbench::usd::MeshBuffers buf;
		buf.positions.reserve(mm.positions.size());
		for (const auto &p : mm.positions)
			buf.positions.push_back({p[0], p[1], p[2]});
		if (!mm.lods.empty() && !mm.lods[0].indices.empty()) {
			const auto &span = mm.lods[0].indices;
			buf.indices.assign(span.begin(), span.end());
		} else {
			buf.indices = mm.indices;
		}
		meshes.push_back(std::move(buf));
	}

	for (size_t mi = 0; mi < mv.meshes.size(); ++mi) {
		const auto &mm = mv.meshes[mi];
		if (mm.transforms.empty())
			continue;
		for (const auto &xf : mm.transforms) {
			accbench::usd::Instance inst;
			inst.meshIndex = static_cast<int>(mi);
			Matr4fToColumnMajorGl(xf, inst.modelColumnMajor);
			instances.push_back(inst);
		}
	}

	if (instances.empty()) {
		err = "mr-importer: no instances";
		return false;
	}

	std::fprintf(stderr, "mr-importer: %zu meshes, %zu instances\n", meshes.size(), instances.size());
	std::fflush(stderr);
	return true;
}

} // namespace accbench::mrimp
