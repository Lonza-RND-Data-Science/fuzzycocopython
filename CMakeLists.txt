cmake_minimum_required(VERSION 3.23)
project(fuzzycocopython LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(POLICY CMP0148)
  cmake_policy(SET CMP0148 NEW)
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

# --- turn off tests/bench/examples in the subproject (cover common flags) ---
set(BUILD_TESTING OFF CACHE BOOL "Disable testing in subprojects" FORCE)
set(ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(FUZZYCOCO_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(FUZZYCOCO_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(FUZZYCOCO_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(FUZZYCOCO_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)

if(NOT MSVC)
  add_compile_options(-Wno-error)
endif()

# bring in fuzzycoco
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/fuzzycoco EXCLUDE_FROM_ALL)
add_subdirectory(bindings)
