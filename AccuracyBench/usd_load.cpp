#include "usd_load.hpp"

#include "composition.hh"
#include "io-util.hh"
#include "prim-types.hh"
#include "tinyusdz.hh"
#include "tydra/render-data.hh"
#include "usdGeom.hh"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#if !defined(_WIN32)
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace accbench::usd {
namespace {

static void UsdProgress(const char *msg) {
	std::fprintf(stderr, "%s\n", msg);
	std::fflush(stderr);
}

// Tydra fails on some game assets (multi-root Skeleton after heavy composition).
// For occlusion / bind-pose geometry we only need static points; drop skin inputs.
static void StripSkelSkinForBindPose(tinyusdz::Prim &prim) {
	if (tinyusdz::GeomMesh *mesh = prim.get_data().as<tinyusdz::GeomMesh>()) {
		mesh->skeleton = nonstd::nullopt;
		mesh->blendShapeTargets = nonstd::nullopt;
		static constexpr const char *kSkelPrimvars[] = {"skel:jointIndices", "skel:jointWeights", "skel:geomBindTransform",
		    "skel:joints"};
		for (const char *name : kSkelPrimvars) {
			const std::string base = std::string("primvars:") + name;
			mesh->props.erase(base);
			mesh->props.erase(base + ":indices");
		}
	}
	for (auto &ch : prim.children())
		StripSkelSkinForBindPose(ch);
}

static void StripSkelSkinForBindPose(tinyusdz::Stage &stage) {
	for (auto &root : stage.root_prims())
		StripSkelSkinForBindPose(root);
}

#if !defined(_WIN32)
static bool ShellSingleQuotedPath(const std::string &path, std::string &out) {
	out.clear();
	out.push_back('\'');
	for (char c : path) {
		if (c == '\'')
			return false;
		out.push_back(c);
	}
	out.push_back('\'');
	return true;
}

// Fallback when TinyUSDZ composition still yields no meshes (pxr usdcat flatten is more complete).
static bool TryLoadViaUsdcatFlatten(const std::string &filepath, tinyusdz::Stage *out_stage, std::string &warn,
    std::string &err) {
	std::string qin, qout;
	if (!ShellSingleQuotedPath(filepath, qin))
		return false;

	UsdProgress(
	    "USD: running `usdcat --flatten` (Pixar USD). For large prefabs this alone can take 1–3 minutes; "
	    "the next parse/Tydra step often adds several more minutes. Window opens only after this finishes.");

	const char kTpl[] = "/tmp/accbench_usd_flat_XXXXXX.usda";
	std::vector<char> tmpl(kTpl, kTpl + sizeof(kTpl));
	int fd = mkstemps(tmpl.data(), 5);
	if (fd < 0)
		return false;
	close(fd);

	std::string outpath(tmpl.data());
	if (!ShellSingleQuotedPath(outpath, qout)) {
		unlink(outpath.c_str());
		return false;
	}

	const std::string cmd = "usdcat --flatten " + qin + " -o " + qout + " 2>/dev/null";
	const int rc = std::system(cmd.c_str());
	const bool ok = rc != -1 && WIFEXITED(rc) && WEXITSTATUS(rc) == 0;
	if (!ok) {
		unlink(outpath.c_str());
		return false;
	}

	tinyusdz::USDLoadOptions load_opts;
	std::string w, e;
	UsdProgress("USD: parsing flattened .usda with TinyUSDZ (large file = CPU-bound, no progress bar)…");
	const bool loaded = tinyusdz::LoadUSDFromFile(outpath, out_stage, &w, &e);
	unlink(outpath.c_str());
	if (!w.empty()) {
		if (!warn.empty())
			warn += "\n";
		warn += w;
	}
	if (!loaded) {
		if (!e.empty()) {
			if (!err.empty())
				err += "\n";
			err += "usdcat flatten load: " + e;
		}
		return false;
	}
	return true;
}
#endif

static void Matrix4dToColumnMajorGl(const tinyusdz::value::matrix4d &M, float out[16]) {
	for (int c = 0; c < 4; ++c)
		for (int r = 0; r < 4; ++r)
			out[c * 4 + r] = static_cast<float>(M.m[r][c]);
}

static void CollectInstances(const tinyusdz::tydra::Node &node, const tinyusdz::tydra::RenderScene &scene,
    std::vector<Instance> &instances) {
	using NT = tinyusdz::tydra::NodeType;
	if (node.nodeType == NT::Mesh && node.id >= 0 &&
	    static_cast<size_t>(node.id) < scene.meshes.size()) {
		Instance inst;
		inst.meshIndex = node.id;
		Matrix4dToColumnMajorGl(node.global_matrix, inst.modelColumnMajor);
		instances.push_back(inst);
	}
	for (const auto &ch : node.children)
		CollectInstances(ch, scene, instances);
}

static bool BuildMeshBuffers(const tinyusdz::tydra::RenderMesh &rm, MeshBuffers &out, std::string &err) {
	const auto &idx = rm.faceVertexIndices();
	const auto &counts = rm.faceVertexCounts();
	if (rm.points.empty() || idx.empty()) {
		err = "empty mesh points or indices";
		return false;
	}
	if (!counts.empty()) {
		for (uint32_t fc : counts) {
			if (fc != 3) {
				err = "non-triangle face after triangulation (faceVertexCounts != 3)";
				return false;
			}
		}
	} else if (idx.size() % 3 != 0) {
		err = "index count not multiple of 3";
		return false;
	}
	out.positions.reserve(rm.points.size());
	for (const auto &p : rm.points)
		out.positions.push_back({p[0], p[1], p[2]});
	out.indices.assign(idx.begin(), idx.end());
	return true;
}

// Like tusdcat --flatten (non-USDZ): resolve subLayers, then references/payload/inherits/variants.
static bool TryFlattenUsdStage(const std::string &filepath, tinyusdz::Stage *out_stage, std::string &warn,
    std::string &err) {
	tinyusdz::Layer root_layer;
	tinyusdz::USDLoadOptions load_opts;
	if (!tinyusdz::LoadLayerFromFile(filepath, &root_layer, &warn, &err, load_opts))
		return false;

	const std::string base_dir = tinyusdz::io::GetBaseDir(filepath);
	tinyusdz::AssetResolutionResolver resolver;
	resolver.set_current_working_path(base_dir);
	resolver.set_search_paths({base_dir});

	tinyusdz::Layer src_layer = std::move(root_layer);
	{
		tinyusdz::Layer composited_layer;
		if (!tinyusdz::CompositeSublayers(resolver, src_layer, &composited_layer, &warn, &err))
			return false;
		src_layer = std::move(composited_layer);
	}

	constexpr int kMaxIteration = 128;
	for (int iter = 0; iter < kMaxIteration; ++iter) {
		bool has_unresolved = false;

		if (src_layer.check_unresolved_references()) {
			has_unresolved = true;
			tinyusdz::Layer composited_layer;
			if (!tinyusdz::CompositeReferences(resolver, src_layer, &composited_layer, &warn, &err))
				return false;
			src_layer = std::move(composited_layer);
		}

		if (src_layer.check_unresolved_payload()) {
			has_unresolved = true;
			tinyusdz::Layer composited_layer;
			if (!tinyusdz::CompositePayload(resolver, src_layer, &composited_layer, &warn, &err))
				return false;
			src_layer = std::move(composited_layer);
		}

		if (src_layer.check_unresolved_inherits()) {
			has_unresolved = true;
			tinyusdz::Layer composited_layer;
			if (!tinyusdz::CompositeInherits(src_layer, &composited_layer, &warn, &err))
				return false;
			src_layer = std::move(composited_layer);
		}

		if (src_layer.check_unresolved_variant()) {
			has_unresolved = true;
			tinyusdz::Layer composited_layer;
			if (!tinyusdz::CompositeVariant(src_layer, &composited_layer, &warn, &err))
				return false;
			src_layer = std::move(composited_layer);
		}

		if (!has_unresolved)
			break;
	}

	return tinyusdz::LayerToStage(src_layer, out_stage, &warn, &err);
}

static bool StageToMeshesAndInstances(const char *path, tinyusdz::Stage &stage, std::vector<MeshBuffers> &meshes,
    std::vector<Instance> &instances, std::string &warn, std::string &err) {
	meshes.clear();
	instances.clear();

	StripSkelSkinForBindPose(stage);

	UsdProgress("USD: Tydra ConvertToRenderScene (traverse meshes; can take many minutes on big maps)…");

	tinyusdz::tydra::RenderSceneConverter converter;
	tinyusdz::tydra::RenderSceneConverterEnv env(stage);
	env.mesh_config.triangulate = true;
	env.mesh_config.build_vertex_indices = true;
	env.scene_config.load_texture_assets = false;

	const std::string usd_basedir = tinyusdz::io::GetBaseDir(path);
	env.set_search_paths({usd_basedir});

	if (tinyusdz::IsUSDZ(path)) {
		tinyusdz::USDZAsset usdz_asset;
		std::string w, e;
		if (!tinyusdz::ReadUSDZAssetInfoFromFile(path, &usdz_asset, &w, &e)) {
			err = e.empty() ? "ReadUSDZAssetInfoFromFile failed" : e;
			return false;
		}
		if (!w.empty()) {
			if (!warn.empty())
				warn += "\n";
			warn += w;
		}
		tinyusdz::AssetResolutionResolver arr;
		if (!tinyusdz::SetupUSDZAssetResolution(arr, &usdz_asset)) {
			err = "SetupUSDZAssetResolution failed";
			return false;
		}
		env.asset_resolver = arr;
	}

	tinyusdz::tydra::RenderScene render_scene;
	if (!converter.ConvertToRenderScene(env, &render_scene)) {
		err = converter.GetError();
		if (!converter.GetWarning().empty()) {
			if (!warn.empty())
				warn += "\n";
			warn += converter.GetWarning();
		}
		return false;
	}
	if (!converter.GetWarning().empty()) {
		if (!warn.empty())
			warn += "\n";
		warn += converter.GetWarning();
	}

	meshes.resize(render_scene.meshes.size());
	for (size_t i = 0; i < render_scene.meshes.size(); ++i) {
		std::string me;
		if (!BuildMeshBuffers(render_scene.meshes[i], meshes[i], me)) {
			err = "mesh " + std::to_string(i) + ": " + me;
			return false;
		}
	}

	for (const auto &root : render_scene.nodes)
		CollectInstances(root, render_scene, instances);

	return !instances.empty();
}

} // namespace

bool LoadStage(const char *path, std::vector<MeshBuffers> &meshes, std::vector<Instance> &instances,
    std::string &warn, std::string &err) {
	warn.clear();
	err.clear();
	meshes.clear();
	instances.clear();

	if (!path || !path[0]) {
		err = "empty path";
		return false;
	}
	if (!tinyusdz::IsUSD(path)) {
		err = "not a USD path (TinyUSDZ IsUSD)";
		return false;
	}

	const std::string filepath(path);
	std::string w, e;
	tinyusdz::Stage stage;

	UsdProgress("USD: TinyUSDZ LoadUSDFromFile…");
	if (!tinyusdz::LoadUSDFromFile(filepath, &stage, &w, &e)) {
		err = e.empty() ? "LoadUSDFromFile failed" : e;
		warn = w;
		return false;
	}
	if (!w.empty())
		warn = std::move(w);

	if (StageToMeshesAndInstances(path, stage, meshes, instances, warn, err))
		return true;

	// Typical for game USD: geometry lives behind references/payloads (see tusdcat --flatten).
	if (!tinyusdz::IsUSDZ(filepath)) {
		UsdProgress("USD: first pass had no drawable meshes; TinyUSDZ composition (references/payloads), often ~30–120 s…");
		tinyusdz::Stage flat;
		std::string fw = warn, fe;
		if (TryFlattenUsdStage(filepath, &flat, fw, fe) &&
		    StageToMeshesAndInstances(path, flat, meshes, instances, fw, fe)) {
			warn = std::move(fw);
			return true;
		}
		if (!fe.empty()) {
			if (!warn.empty())
				warn += "\n";
			warn += "composition flatten: " + fe;
		}
	}

#if !defined(_WIN32)
	if (!tinyusdz::IsUSDZ(filepath)) {
		UsdProgress(
		    "USD: still no meshes or load failed; trying pxr `usdcat` flatten fallback (needs usdcat in PATH)…");
		tinyusdz::Stage pxr_flat;
		std::string fw = warn, fe;
		if (TryLoadViaUsdcatFlatten(filepath, &pxr_flat, fw, fe) &&
		    StageToMeshesAndInstances(path, pxr_flat, meshes, instances, fw, fe)) {
			warn = std::move(fw);
			return true;
		}
		if (!fe.empty()) {
			if (!warn.empty())
				warn += "\n";
			warn += fe;
		}
	}
#endif

	if (err.empty())
		err = "no Mesh instances in RenderScene (TinyUSDZ composition incomplete, skel conversion failed, or missing "
		      "usdcat in PATH for pxr flatten fallback)";
	return false;
}

} // namespace accbench::usd
