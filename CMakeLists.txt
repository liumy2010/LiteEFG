cmake_minimum_required(VERSION 3.4...3.18)
project(LiteEFG)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(pybind11)
file(GLOB_RECURSE all_src
     "LiteEFG/src/*.h"
     "LiteEFG/src/*.cpp"
)

message(${all_src})
pybind11_add_module(_LiteEFG ${all_src})
target_include_directories(_LiteEFG PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/LiteEFG/src)
target_compile_options(_LiteEFG PRIVATE -O3)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(_LiteEFG
                           PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
