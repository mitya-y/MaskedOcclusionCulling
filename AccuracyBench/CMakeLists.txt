cmake_minimum_required(VERSION 3.27)
project(AccuracyBench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)

# mr-math headers need Vc (SimdImpl); upstream mr-math CMake references Vc::Vc but 4J Vc provides target "Vc".
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

add_library(mr-math-lib INTERFACE)
target_include_directories(mr-math-lib INTERFACE ${mr_math_fc_SOURCE_DIR}/include)
target_link_libraries(mr-math-lib INTERFACE Vc)
target_compile_features(mr-math-lib INTERFACE cxx_std_23)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/.. ./MaskedOcclusionCulling)

# OpenGL preview + readback use GL clip/screen space; MOC defaults to D3D-style Y unless USE_D3D=0.
target_compile_definitions(MaskedOcclusionCulling PRIVATE USE_D3D=0)

add_executable(AccuracyBench
	main.cpp
	cgltf_impl.cpp
	usd_load.cpp
)
# Per-source CXX_STANDARD is ignored by some generators; force language version for TinyUSDZ headers.
set_source_files_properties(usd_load.cpp PROPERTIES COMPILE_OPTIONS "-std=c++17")

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
