# Consumer Conan recipe for AccuracyBench with optional OpenUSD via mr-importer (4J).
#
# Prerequisites:
#   - C++23 (set compiler.cppstd in your Conan profile, e.g. gnu23 or 23).
#   - Recipes for 4J packages (mr-importer pulls fastgltf/0.9.0-4j, mr-math, openusd, …).
#     Use the remote / local cache where you already build mr-importer, or:
#       git clone https://github.com/4J-company/mr-importer && cd mr-importer && conan create . --version=3.4.0
#
# Typical workflow (from AccuracyBench/, same folder as this file):
#   conan install . --output-folder=build --build=missing -s compiler.cppstd=gnu23
#   cmake --preset conan-release
#   cmake --build --preset conan-release
#
# Without presets, point CMake at the toolchain (Linux Release example):
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
#     -DCMAKE_TOOLCHAIN_FILE=build/Release/generators/conan_toolchain.cmake
#   cmake --build build
#
# Use one and the same --output-folder as your CMake build tree so ACCURACYBENCH_CONAN_GENERATORS
# autodetection finds *-data.cmake; or pass -DACCURACYBENCH_CONAN_GENERATORS=.../generators if needed.
#
# The generated toolchain sets ACCURACYBENCH_USE_MR_IMPORTER from option mr_importer.
# For USD plugins at runtime you may need (same as mr-importer / OpenUSD):
#   export PXR_PLUGINPATH="…"   # or use the path under openusd package lib/usd
#
# If cmake says "Duplicate preset: conan-release", remove CMakeUserPresets.json and run
# `conan install` again (or rely on generate() below, which rewrites a single include).
#
import json
from pathlib import Path

from conan import ConanFile
from conan.tools.cmake import CMakeDeps, CMakeToolchain, cmake_layout
from conan.tools.build import check_min_cppstd


class AccuracyBenchConan(ConanFile):
    settings = "os", "compiler", "build_type", "arch"

    options = {
        "mr_importer": [True, False],
    }
    default_options = {
        "mr_importer": True,
    }

    def requirements(self):
        # Always: headers + Vc via Conan (avoids FetchContent mr-math + flaky CPM Vc).
        self.requires("mr-math/[>1.1.4]")
        if self.options.mr_importer:
            self.requires("mr-importer/3.4.0")

    def build_requirements(self):
        self.tool_requires("cmake/[>=3.27]")

    def layout(self):
        # Default cmake_layout uses build_folder="build". With `conan install --output-folder=build` that
        # yields build/build/Release. Use an empty base so layout is <output-folder>/Release/… .
        cmake_layout(self, build_folder="")

    def validate(self):
        check_min_cppstd(self, "23")

    def _rewrite_cmake_user_presets_single_include(self) -> None:
        """Conan can accumulate extra `include` entries in CMakeUserPresets.json; CMake rejects duplicate preset names."""
        bf = Path(self.build_folder)
        bt = str(self.settings.build_type)
        gen_presets = None
        for candidate in (
            bf / bt / "generators" / "CMakePresets.json",
            bf / "build" / bt / "generators" / "CMakePresets.json",
            bf / "generators" / "CMakePresets.json",
        ):
            if candidate.is_file():
                gen_presets = candidate
                break
        if gen_presets is None:
            return

        src = Path(self.source_folder)
        try:
            include_path = gen_presets.relative_to(src).as_posix()
        except ValueError:
            include_path = gen_presets.as_posix()

        user = {
            "version": 4,
            "vendor": {"conan": {}},
            "include": [include_path],
        }
        (src / "CMakeUserPresets.json").write_text(json.dumps(user, indent=4) + "\n", encoding="utf-8")

    def generate(self):
        tc = CMakeToolchain(self)
        tc.cache_variables["ACCURACYBENCH_USE_MR_IMPORTER"] = (
            "ON" if self.options.mr_importer else "OFF"
        )
        if self.options.mr_importer:
            openusd = self.dependencies.get("openusd")
            if openusd is not None:
                pkg = Path(openusd.package_folder)
                for sub in ("lib/usd", "lib64/usd"):
                    candidate = pkg / sub
                    plug = candidate / "sdf" / "resources" / "plugInfo.json"
                    if plug.is_file():
                        tc.cache_variables["MR_IMPORTER_PXR_USD_PLUGIN_ROOT"] = str(candidate).replace(
                            "\\", "/"
                        )
                        break
        tc.generate()
        CMakeDeps(self).generate()
        self._rewrite_cmake_user_presets_single_include()
