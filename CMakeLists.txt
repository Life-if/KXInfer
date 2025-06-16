cmake_minimum_required(VERSION 3.31)
project(KXInfer)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
include(cmake/cuda.cmake)

find_package(glog REQUIRED)
find_package(GTest REQUIRED)
find_package(Armadillo REQUIRED)

aux_source_directory(src/base/ DIR_BASE)
aux_source_directory(src/op/ DIR_OP)
aux_source_directory(src/tensor/ DIR_TENSOR)
aux_source_directory(src/op/kernel/cpu DIR_KERNEL_CPU)
aux_source_directory(src/op/kernel/cuda DIR_KERNEL_CUDA)
aux_source_directory(src/op/kernel DIR_KERNEL)


add_library(KXInfer SHARED ${DIR_BASE} ${DIR_OP} ${DIR_TENSOR} ${DIR_KERNEL} ${DIR_KERNEL_CPU} ${DIR_KERNEL_CUDA})
#add_executable(KXInfer main.cpp)

target_link_libraries(KXInfer sentencepiece glog::glog gtest gtest_main pthread cudart armadillo)
target_link_directories(KXInfer PUBLIC ${CMAKE_CUDA_COMPILER_LIBRARY_ROOT}/lib64)

target_include_directories(KXInfer PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(KXInfer PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(KXInfer PUBLIC ${Armadillo_INCLUDE_DIR})
target_include_directories(KXInfer PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

target_include_directories(KXInfer PUBLIC ${PROJECT_SOURCE_DIR}/src)

set_target_properties(KXInfer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_subdirectory(test)

