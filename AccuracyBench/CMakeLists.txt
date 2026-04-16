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
)

target_include_directories(AccuracyBench PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(AccuracyBench PRIVATE
	mr-math-lib
	MaskedOcclusionCulling
	OpenGL::GL
	GLEW::GLEW
	glfw
)

target_compile_definitions(AccuracyBench PRIVATE USE_D3D=0)

if(NOT WIN32)
	target_link_libraries(AccuracyBench PRIVATE pthread dl)
endif()
