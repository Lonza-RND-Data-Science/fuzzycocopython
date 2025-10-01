cmake_minimum_required(VERSION 3.16)
project(fuzzycoco_unit_tests VERSION 1.0 LANGUAGES C CXX)

# we require at least C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
if (MSVC)
  set(CMAKE_CXX_FLAGS "/Zi /W4")
else()
  # GCC/Clang flags
  # set(CMAKE_CXX_FLAGS "-O2 -g -Wall")
endif()

include_directories(${PROJECT_SOURCE_DIR}/../../src)

# GoogleTest setup

# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
  cmake_policy(SET CMP0135 NEW)
endif()

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
)
# force static linking of Gtest
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)    # <--
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)
include(GoogleTest)
enable_testing()

# ====================== TESTS ========================
# Function to add Google Test cases
function(add_gtest name)
    set(test_name "${name}_test")
    # Create executable for the test
    add_executable(${test_name} ${test_name}.cpp)

    # Link GoogleTest and any other necessary libraries (libfuzzycoco in this case)
    target_link_libraries(${test_name} GTest::gtest_main fuzzycoco)

    # Discover and register the test with CTest
    gtest_discover_tests(${test_name})
endfunction()

add_gtest(bitarray)
add_gtest(coevolution_engine)
add_gtest(crossover_method)
add_gtest(dataframe)
add_gtest(digest)
add_gtest(discretizer)
add_gtest(evolution_engine)
add_gtest(file_utils)
add_gtest(fuzzy_coco)
add_gtest(fuzzy_coco_codec)
add_gtest(fuzzy_coco_engine)
add_gtest(fuzzy_coco_features_weights)
add_gtest(fuzzy_coco_fitness)
add_gtest(fuzzy_coco_params)
add_gtest(fuzzy_default_rules)
add_gtest(fuzzy_rules)
add_gtest(fuzzy_set)
add_gtest(fuzzy_system)
add_gtest(fuzzy_system_fitness)
add_gtest(fuzzy_system_metrics)
add_gtest(fuzzy_variable)
add_gtest(genome_codec)
add_gtest(logging_logger)
add_gtest(mutation_method)
add_gtest(named_list)
add_gtest(selection_method)
add_gtest(random_generator)
add_gtest(types)
add_gtest(string_utils)