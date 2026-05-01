conan install . --output-folder=build --build=missing -s compiler.cppstd=gnu23
cmake --preset conan-release
cmake --build --preset conan-release -j16



