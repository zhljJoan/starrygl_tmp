cmake_minimum_required(VERSION 3.15)

project(starry_unigraph_bts_sampler VERSION 0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)
find_package(Torch REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${Python3_INCLUDE_DIRS})
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/sampler/include")

link_libraries(OpenMP::OpenMP_CXX)

add_library(starry_unigraph_bts_sampler SHARED sampler/export.cpp)
target_compile_options(starry_unigraph_bts_sampler PRIVATE -O3)
target_compile_definitions(starry_unigraph_bts_sampler PRIVATE -DTORCH_EXTENSION_NAME=libstarrygl_sampler)
target_link_libraries(starry_unigraph_bts_sampler PRIVATE ${TORCH_LIBRARIES})

find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
if(TORCH_PYTHON_LIBRARY)
  target_link_libraries(starry_unigraph_bts_sampler PRIVATE ${TORCH_PYTHON_LIBRARY})
endif()
