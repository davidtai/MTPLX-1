cmake_minimum_required(VERSION 3.27)

project(mtplx_native_qsa_ext LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_SHARED_LIBS "Build extensions as a shared library" ON)

find_package(
  Python 3.11
  COMPONENTS Interpreter Development.Module
  REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE MLX_ROOT)
find_package(MLX CONFIG REQUIRED)

add_library(mtplx_native_qsa_ext)
target_sources(
  mtplx_native_qsa_ext
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/sparse_gqa/qsa_sparse_gqa.cpp)
target_include_directories(mtplx_native_qsa_ext PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(mtplx_native_qsa_ext PUBLIC mlx)

if(MLX_BUILD_METAL)
  mlx_build_metallib(
    TARGET
    mtplx_native_qsa_ext_metallib
    TITLE
    mtplx_native_qsa_ext
    SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/sparse_gqa/qsa_sparse_gqa.metal
    INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}
    ${MLX_INCLUDE_DIRS}
    DEPS
    ${CMAKE_CURRENT_LIST_DIR}/sparse_gqa/steel_qsa_sparse_gqa.h
    ${CMAKE_CURRENT_LIST_DIR}/sparse_gqa/qsa_sparse_gqa_params.h
    OUTPUT_DIRECTORY
    ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

  add_dependencies(mtplx_native_qsa_ext mtplx_native_qsa_ext_metallib)
endif()

nanobind_add_module(
  _ext
  NB_STATIC
  STABLE_ABI
  LTO
  NOMINSIZE
  NB_DOMAIN
  mlx
  ${CMAKE_CURRENT_LIST_DIR}/bindings.cpp)
target_link_libraries(_ext PRIVATE mtplx_native_qsa_ext)

if(BUILD_SHARED_LIBS)
  target_link_options(_ext PRIVATE -Wl,-rpath,@loader_path)
endif()
