add_library(tungsten_gp_cuda STATIC
    cuda/SeSplat1D.cu
    cuda/RaySegmentBatch.cu
    cuda/RaySegmentBatch.cpp)

target_include_directories(tungsten_gp_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(tungsten_gp_cuda PUBLIC CUDA::cudart)
set_target_properties(tungsten_gp_cuda PROPERTIES
    CXX_STANDARD 17
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON)

if (OpenVDB_FOUND AND OpenEXR_FOUND AND TBB_FOUND)
    add_library(tungsten_gp_cuda_vdb STATIC VdbMeanGrid.cpp)
    target_include_directories(tungsten_gp_cuda_vdb PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(tungsten_gp_cuda_vdb PUBLIC tungsten_gp_cuda ${core_libs})

    add_library(tungsten_gp_cuda_pathtracer STATIC PathTracerSeBatchBackend.cpp)
    target_include_directories(tungsten_gp_cuda_pathtracer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(tungsten_gp_cuda_pathtracer PUBLIC tungsten_gp_cuda_vdb)
endif()

option(TUNGSTEN_CUDA_BUILD_TESTS "Build the standalone GPIS CUDA test" ON)
if (TUNGSTEN_CUDA_BUILD_TESTS)
    add_executable(tungsten_gp_cuda_test tests/SeSplat1DTest.cpp)
    target_link_libraries(tungsten_gp_cuda_test PRIVATE tungsten_gp_cuda)
    add_test(NAME tungsten_gp_cuda_se_splat_1d COMMAND tungsten_gp_cuda_test)
    set_tests_properties(tungsten_gp_cuda_se_splat_1d PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(tungsten_gp_cuda_ray_batch_test tests/RaySegmentBatchTest.cpp)
    target_link_libraries(tungsten_gp_cuda_ray_batch_test PRIVATE tungsten_gp_cuda)
    add_test(NAME tungsten_gp_cuda_ray_batch COMMAND tungsten_gp_cuda_ray_batch_test)
    set_tests_properties(tungsten_gp_cuda_ray_batch PROPERTIES SKIP_RETURN_CODE 77)

    if (TARGET tungsten_gp_cuda_vdb AND
        EXISTS "${PROJECT_SOURCE_DIR}/gpis-scenes/shaderball_ply/scene.json")
        add_executable(tungsten_gp_cuda_vdb_test tests/VdbMeanGridTest.cpp)
        target_link_libraries(tungsten_gp_cuda_vdb_test PRIVATE tungsten_gp_cuda_vdb)
        add_test(NAME tungsten_gp_cuda_vdb
                 COMMAND tungsten_gp_cuda_vdb_test
                         "${PROJECT_SOURCE_DIR}/gpis-scenes/shaderball_ply/scene.json")
        set_tests_properties(tungsten_gp_cuda_vdb PROPERTIES
            SKIP_RETURN_CODE 77
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/gpis-scenes/shaderball_ply")
        add_executable(tungsten_gp_wavefront_test tests/WavefrontPathTracerTest.cpp)
        target_link_libraries(tungsten_gp_wavefront_test PRIVATE tungsten_gp_cuda_pathtracer)
        add_test(NAME tungsten_gp_wavefront
                 COMMAND tungsten_gp_wavefront_test
                         "${PROJECT_SOURCE_DIR}/gpis-scenes/shaderball_ply/scene.json")
        set_tests_properties(tungsten_gp_wavefront PROPERTIES
            TIMEOUT 120
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/gpis-scenes/shaderball_ply")
    endif()
endif()
