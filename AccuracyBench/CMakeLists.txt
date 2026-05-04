cmake_minimum_required(VERSION 3.27)
project(AccuracyBench LANGUAGES CXX)

# Conan CMakeDeps sometimes leaves INTERFACE_INCLUDE_DIRECTORIES empty (e.g. editable packages).
# Read PACKAGE_FOLDER_* from generators/mr-*-data.cmake and use <root>/include when present.
function(accbench_conan_include_from_generators_data package_slug out_var)
	set(_result "")
	if(CMAKE_BUILD_TYPE)
		string(TOUPPER "${CMAKE_BUILD_TYPE}" _btu)
		set(_gen_roots "")
		if(ACCURACYBENCH_CONAN_GENERATORS)
			list(APPEND _gen_roots "${ACCURACYBENCH_CONAN_GENERATORS}")
		endif()
		list(
			APPEND
			_gen_roots
			"${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/generators"
			"${CMAKE_BINARY_DIR}/generators"
		)
		foreach(_gen_root IN LISTS _gen_roots)
			if(NOT EXISTS "${_gen_root}")
				continue()
			endif()
			file(GLOB _data "${_gen_root}/${package_slug}-*-data.cmake")
			if(NOT _data)
				continue()
			endif()
			list(GET _data 0 _data_file)
			file(STRINGS "${_data_file}" _lines REGEX "PACKAGE_FOLDER_${_btu}")
			foreach(_line IN LISTS _lines)
				if(_line MATCHES "\\\"([^\"]+)\\\"\\)")
					set(_root "${CMAKE_MATCH_1}")
					if(EXISTS "${_root}/include")
						set(_result "${_root}/include")
						break()
					endif()
				endif()
			endforeach()
			if(_result)
				break()
			endif()
		endforeach()
	endif()
	set(${out_var} "${_result}" PARENT_SCOPE)
endfunction()

macro(accbench_prepend_conan_generators_to_prefix)
	if(ACCURACYBENCH_CONAN_GENERATORS)
		list(PREPEND CMAKE_PREFIX_PATH "${ACCURACYBENCH_CONAN_GENERATORS}")
	endif()
	foreach(
		_d
		IN
		ITEMS
			"${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/generators"
			"${CMAKE_BINARY_DIR}/generators"
			"${CMAKE_BINARY_DIR}/Release/generators"
			"${CMAKE_BINARY_DIR}/Debug/generators"
	)
		if(IS_DIRECTORY "${_d}")
			list(PREPEND CMAKE_PREFIX_PATH "${_d}")
		endif()
	endforeach()
endmacro()

# Fast USD path via 4J mr-importer (OpenUSD). Requires Conan (or equivalent) to provide
# mr-importer and all of its dependencies (pxr, meshoptimizer, fastgltf, …).
option(
	ACCURACYBENCH_USE_MR_IMPORTER
	"Link mr-importer for USD (OpenUSD). Requires Conan (or prefix) for mr-math + mr-importer. Conan preset turns this ON."
	ON
)
set(
	ACCURACYBENCH_CONAN_GENERATORS ""
	CACHE
	PATH
	"Conan CMakeDeps folder (…/generators with *-config.cmake and *-data.cmake). Optional if CMAKE_BINARY_DIR matches conan install --output-folder."
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)

accbench_prepend_conan_generators_to_prefix()

set(_accbench_mr_math_from_conan FALSE)
find_package(mr-math CONFIG QUIET)
if(TARGET mr-math::mr-math)
	set(_accbench_mr_math_from_conan TRUE)
elseif(ACCURACYBENCH_USE_MR_IMPORTER)
	message(
		FATAL_ERROR
		"ACCURACYBENCH_USE_MR_IMPORTER=ON but CMake did not find package mr-math (Conan CMakeDeps). "
		"Run `conan install` for AccuracyBench (see conanfile.py) or set CMAKE_PREFIX_PATH / "
		"ACCURACYBENCH_CONAN_GENERATORS to the Conan generators folder."
	)
else()
	# No Conan mr-math: FetchContent + CPM Vc (fragile on some toolchains).
	file(
		DOWNLOAD
		https://github.com/cpm-cmake/CPM.cmake/releases/download/v0.40.2/CPM.cmake
		${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake
		EXPECTED_HASH SHA256=c8cdc32c03816538ce22781ed72964dc864b2a34a310d3b7104812a5ca2d835d
	)
	include(${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake)

	CPMFindPackage(
		NAME Vc
		GITHUB_REPOSITORY 4J-company/Vc
		GIT_TAG 1.4
	)

	include(FetchContent)
	FetchContent_Declare(
		mr_math_fc
		GIT_REPOSITORY https://github.com/4J-company/mr-math.git
		GIT_TAG master
		GIT_SHALLOW TRUE
	)
	FetchContent_GetProperties(mr_math_fc)
	if(NOT mr_math_fc_POPULATED)
		FetchContent_Populate(mr_math_fc)
	endif()

	add_library(mr-math-lib INTERFACE)
	target_include_directories(mr-math-lib INTERFACE ${mr_math_fc_SOURCE_DIR}/include)
	target_link_libraries(mr-math-lib INTERFACE Vc)
	target_compile_features(mr-math-lib INTERFACE cxx_std_23)
endif()

if(_accbench_mr_math_from_conan)
	add_library(mr-math-lib ALIAS mr-math::mr-math)
endif()

if(ACCURACYBENCH_USE_MR_IMPORTER)
	find_package(mr-utils CONFIG QUIET)
	find_package(mr-importer CONFIG REQUIRED)
	if(NOT TARGET mr-importer::mr-importer-lib AND NOT TARGET mr-importer::mr-importer AND NOT TARGET mr-importer-lib)
		message(
			FATAL_ERROR
			"ACCURACYBENCH_USE_MR_IMPORTER: expected mr-importer CMake package "
			"(mr-importer::mr-importer-lib from cmake install, or mr-importer::mr-importer from Conan CMakeDeps)."
		)
	endif()
endif()

include(FetchContent)
set(TINYUSDZ_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(TINYUSDZ_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(TINYUSDZ_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(TINYUSDZ_NO_WERROR ON CACHE BOOL "" FORCE)

FetchContent_Declare(
	tinyusdz_fc
	GIT_REPOSITORY https://github.com/lighttransport/tinyusdz.git
	GIT_TAG release
	GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(tinyusdz_fc)
# TinyUSDZ nonstd::expected conflicts with C++23 std::expected on some libstdc++.
if(TARGET tinyusdz_static)
	set_target_properties(tinyusdz_static PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
	if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
		target_compile_options(tinyusdz_static PRIVATE -include
			${CMAKE_CURRENT_SOURCE_DIR}/tinyusdz_forced_stdint.h)
	endif()
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/.. ./MaskedOcclusionCulling)

# OpenGL preview + readback use GL clip/screen space; MOC defaults to D3D-style Y unless USE_D3D=0.
target_compile_definitions(MaskedOcclusionCulling PRIVATE USE_D3D=0)

add_executable(AccuracyBench
	main.cpp
	dop26.cpp
	cgltf_impl.cpp
	usd_load.cpp
)
# Per-source CXX_STANDARD is ignored by some generators; force language version for TinyUSDZ headers.
set_source_files_properties(usd_load.cpp PROPERTIES COMPILE_OPTIONS "-std=c++17")

if(ACCURACYBENCH_USE_MR_IMPORTER)
	accbench_conan_include_from_generators_data("mr-math" _accbench_inc_mr_math)
	accbench_conan_include_from_generators_data("mr-utils" _accbench_inc_mr_utils)
	accbench_conan_include_from_generators_data("Vc" _accbench_inc_vc)
	accbench_conan_include_from_generators_data("libassert" _accbench_inc_libassert)
	accbench_conan_include_from_generators_data("cpptrace" _accbench_inc_cpptrace)
	if(_accbench_inc_mr_math)
		target_include_directories(AccuracyBench PRIVATE "${_accbench_inc_mr_math}")
	endif()
	if(_accbench_inc_mr_utils)
		target_include_directories(AccuracyBench PRIVATE "${_accbench_inc_mr_utils}")
	endif()
	if(_accbench_inc_vc)
		target_include_directories(AccuracyBench PRIVATE "${_accbench_inc_vc}")
	endif()
	if(_accbench_inc_libassert)
		target_include_directories(AccuracyBench PRIVATE "${_accbench_inc_libassert}")
	endif()
	if(_accbench_inc_cpptrace)
		target_include_directories(AccuracyBench PRIVATE "${_accbench_inc_cpptrace}")
	endif()
	target_sources(AccuracyBench PRIVATE mr_import_usd.cpp)
	if(TARGET mr-importer::mr-importer-lib)
		set(_accbench_mr_import_tgt mr-importer::mr-importer-lib)
	elseif(TARGET mr-importer::mr-importer)
		set(_accbench_mr_import_tgt mr-importer::mr-importer)
	elseif(TARGET mr-importer-lib)
		set(_accbench_mr_import_tgt mr-importer-lib)
	else()
		message(FATAL_ERROR "ACCURACYBENCH_USE_MR_IMPORTER: no mr-importer link target found")
	endif()
	target_link_libraries(AccuracyBench PRIVATE ${_accbench_mr_import_tgt})
	if(TARGET mr-utils::mr-utils)
		target_link_libraries(AccuracyBench PRIVATE mr-utils::mr-utils)
	endif()
	# Transitive usage requirements from IMPORTED targets do not always propagate to executables here.
	if(TARGET Vc::Vc)
		target_link_libraries(AccuracyBench PRIVATE Vc::Vc)
	endif()
	if(TARGET libassert::assert)
		target_link_libraries(AccuracyBench PRIVATE libassert::assert)
	endif()
	if(TARGET cpptrace::cpptrace)
		target_link_libraries(AccuracyBench PRIVATE cpptrace::cpptrace)
	endif()
	target_compile_definitions(AccuracyBench PRIVATE ACCBENCH_HAVE_MR_IMPORTER=1)
endif()

if(_accbench_mr_math_from_conan AND NOT ACCURACYBENCH_USE_MR_IMPORTER)
	accbench_conan_include_from_generators_data("Vc" _accbench_inc_vc_standalone)
	if(_accbench_inc_vc_standalone)
		target_include_directories(AccuracyBench PRIVATE "${_accbench_inc_vc_standalone}")
	endif()
	if(TARGET Vc::Vc)
		target_link_libraries(AccuracyBench PRIVATE Vc::Vc)
	endif()
endif()

target_include_directories(AccuracyBench PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}
	${tinyusdz_fc_SOURCE_DIR}/src
)

target_link_libraries(AccuracyBench PRIVATE
	mr-math-lib
	tinyusdz::tinyusdz_static
	MaskedOcclusionCulling
	OpenGL::GL
	GLEW::GLEW
	glfw
)

target_compile_definitions(AccuracyBench PRIVATE USE_D3D=0)
target_compile_features(AccuracyBench PRIVATE cxx_std_23)

if(NOT WIN32)
	target_link_libraries(AccuracyBench PRIVATE pthread dl)
endif()
