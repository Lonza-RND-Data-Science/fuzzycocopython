# Build the Python extension
pybind11_add_module(_fuzzycoco_core bindings.cpp)

# Link against whichever target the upstream exports
if(TARGET fuzzycoco_static)
  target_link_libraries(_fuzzycoco_core PRIVATE fuzzycoco_static)
elseif(TARGET fuzzycoco)
  target_link_libraries(_fuzzycoco_core PRIVATE fuzzycoco)
elseif(TARGET fuzzycoco::fuzzycoco)
  target_link_libraries(_fuzzycoco_core PRIVATE fuzzycoco::fuzzycoco)
else()
  message(FATAL_ERROR "Could not find fuzzycoco target from submodule")
endif()

# If the upstream target does NOT export include dirs, add them explicitly.
# First try relying on the target; if that fails, fallback to the submodule path.
get_target_property(_have_includes _fuzzycoco_core INCLUDE_DIRECTORIES)
# Only add a fallback if needed (won't hurt if already present).
target_include_directories(
  _fuzzycoco_core PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../fuzzycoco/src
)

# Place the compiled extension next to your Python package
set_target_properties(_fuzzycoco_core PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../fuzzycocopython
  PREFIX ""
)

install(TARGETS _fuzzycoco_core LIBRARY DESTINATION fuzzycocopython)
