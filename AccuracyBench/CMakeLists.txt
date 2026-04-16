cmake_minimum_required(VERSION 3.16)
project(AccuracyBench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)

find_path(GLM_INCLUDE_DIR glm/glm.hpp
	HINTS ${CMAKE_SYSROOT}/usr/include
	REQUIRED
)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/.. ./MaskedOcclusionCulling)

add_executable(AccuracyBench
	main.cpp
)

target_include_directories(AccuracyBench PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}
	${GLM_INCLUDE_DIR}
)

target_link_libraries(AccuracyBench PRIVATE
	MaskedOcclusionCulling
	OpenGL::GL
	GLEW::GLEW
	glfw
)

if(NOT WIN32)
	target_link_libraries(AccuracyBench PRIVATE pthread dl)
endif()
