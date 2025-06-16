include(../cmake/cuda.cmake)
find_package(GTest REQUIRED)
find_package(glog REQUIRED)
find_package(Armadillo REQUIRED)

set(link_ext_lib glog::glog GTest::gtest)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

aux_source_directory(../test DIR_TEST)
aux_source_directory(../test/test_install DIR_TEST_INSTALL)
aux_source_directory(../test/test_base DIR_TEST_BASE)
aux_source_directory(../test/test_op DIR_TEST_OP)
aux_source_directory(../test/utils DIR_TEST_UTILS)
aux_source_directory(../test/test_tensor DIR_TEST_TENSOR)

add_executable(test_KXInfer ${DIR_TEST} ${DIR_TEST_INSTALL} ${DIR_TEST_BASE} ${DIR_TEST_OP} ${DIR_TEST_TENSOR} ${DIR_TEST_UTILS})

target_link_libraries(test_KXInfer ${link_ext_lib})
target_include_directories(test_KXInfer PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(test_KXInfer PUBLIC ${GTest_INCLUDE_DIR})
target_include_directories(test_KXInfer PUBLIC ${Armadillo_INCLUDE_DIR})
target_include_directories(test_KXInfer PUBLIC ../include)
target_include_directories(test_KXInfer PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
#target_link_directories(test_KXInfer PUBLIC ${PROJECT_SOURCE_DIR}/lib)
target_link_libraries(test_KXInfer KXInfer)

set_target_properties(test_KXInfer PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_target_properties(test_KXInfer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)