cmake_minimum_required(VERSION 3.16)
project(fuzzycoco_debug VERSION 1.0 LANGUAGES C CXX)

# we require at least C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
if (MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zi /W4")
else()
  # GCC/Clang flags
  # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -fprofile-arcs -ftest-coverage -Wall -Werror")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -Werror")
endif()

add_subdirectory(src)
add_subdirectory(tests/unit)
include_directories(${PROJECT_SOURCE_DIR}/src)
