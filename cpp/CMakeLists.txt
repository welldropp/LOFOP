# LOFOP C++ inference SDK.
#
# Builds `lofop` (the C++ API plus the C ABI) from the SDK sources and the
# same native kernels the Python package compiles, so the two runtimes cannot
# drift apart. ONNX Runtime is optional at build time: without it everything
# still compiles and the test suite still runs, and only loading a .onnx by
# path is unavailable.
#
#   cmake -S cpp -B build
#   cmake --build build
#   ctest --test-dir build --output-on-failure
#
# With ONNX Runtime:
#   cmake -S cpp -B build -DLOFOP_WITH_ONNXRUNTIME=ON -DONNXRUNTIME_ROOT=/path/to/onnxruntime

cmake_minimum_required(VERSION 3.16)
project(lofop VERSION 1.2.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(LOFOP_WITH_ONNXRUNTIME "Enable the ONNX Runtime execution backend" OFF)
option(LOFOP_WITH_STB_IMAGE "Decode JPEG/PNG via a vendored stb_image.h" OFF)
option(LOFOP_BUILD_TESTS "Build the SDK test suite" ON)
option(LOFOP_BUILD_EXAMPLES "Build the example programs" ON)
option(BUILD_SHARED_LIBS "Build a shared library instead of static" OFF)

# The kernels live with the Python package; the SDK compiles the same files
# rather than copying them, which is what keeps both languages numerically
# identical.
set(LOFOP_CSRC "${CMAKE_CURRENT_SOURCE_DIR}/../lofop/csrc")

add_library(lofop
  src/image.cpp
  src/detector.cpp
  src/engine_onnxruntime.cpp
  src/capi.cpp
  "${LOFOP_CSRC}/box_ops.cpp"
  "${LOFOP_CSRC}/preprocess.cpp"
)
add_library(lofop::lofop ALIAS lofop)

target_include_directories(lofop PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

if(MSVC)
  target_compile_options(lofop PRIVATE /W4)
else()
  target_compile_options(lofop PRIVATE -Wall -Wextra)
endif()

if(LOFOP_WITH_ONNXRUNTIME)
  # Accept either an ONNXRUNTIME_ROOT pointing at an extracted release, or
  # headers and library already on the default search paths.
  find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    HINTS ${ONNXRUNTIME_ROOT} ENV ONNXRUNTIME_ROOT
    PATH_SUFFIXES include include/onnxruntime include/onnxruntime/core/session
  )
  find_library(ONNXRUNTIME_LIBRARY onnxruntime
    HINTS ${ONNXRUNTIME_ROOT} ENV ONNXRUNTIME_ROOT
    PATH_SUFFIXES lib lib64
  )
  if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR
      "LOFOP_WITH_ONNXRUNTIME=ON but ONNX Runtime was not found. "
      "Download a release from https://github.com/microsoft/onnxruntime/releases "
      "and pass -DONNXRUNTIME_ROOT=/path/to/extracted/onnxruntime.")
  endif()
  target_compile_definitions(lofop PUBLIC LOFOP_WITH_ONNXRUNTIME)
  target_include_directories(lofop PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
  target_link_libraries(lofop PUBLIC ${ONNXRUNTIME_LIBRARY})
  message(STATUS "LOFOP: ONNX Runtime backend enabled (${ONNXRUNTIME_LIBRARY})")
else()
  message(STATUS "LOFOP: building without ONNX Runtime; supply your own Engine to run models")
endif()

if(LOFOP_WITH_STB_IMAGE)
  find_path(STB_IMAGE_INCLUDE_DIR stb_image.h HINTS ${STB_IMAGE_ROOT} ENV STB_IMAGE_ROOT)
  if(NOT STB_IMAGE_INCLUDE_DIR)
    message(FATAL_ERROR "LOFOP_WITH_STB_IMAGE=ON but stb_image.h was not found; "
                        "pass -DSTB_IMAGE_ROOT=/path/containing/stb_image.h")
  endif()
  target_compile_definitions(lofop PRIVATE LOFOP_WITH_STB_IMAGE)
  target_include_directories(lofop PRIVATE ${STB_IMAGE_INCLUDE_DIR})
endif()

if(LOFOP_BUILD_TESTS)
  enable_testing()
  add_executable(lofop_tests tests/test_lofop.cpp)
  target_link_libraries(lofop_tests PRIVATE lofop)
  add_test(NAME lofop_tests COMMAND lofop_tests)
endif()

if(LOFOP_BUILD_EXAMPLES)
  add_executable(detect_image examples/detect_image.cpp)
  target_link_libraries(detect_image PRIVATE lofop)
endif()

include(GNUInstallDirs)
install(TARGETS lofop EXPORT lofopTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT lofopTargets NAMESPACE lofop:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/lofop)
