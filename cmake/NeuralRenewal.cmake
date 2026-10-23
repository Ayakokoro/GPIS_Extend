# Compile the shared scalar sampler, not macrofacet's experiment executable or
# LibTorch/CUDA renderer. A sibling checkout is the default source location.
set(TUNGSTEN_MACROFACET_SOURCE_DIR "${PROJECT_SOURCE_DIR}/../macrofacet" CACHE PATH
    "Macrofacet source checkout containing the causal Renewal sampler")
if (NOT EXISTS "${TUNGSTEN_MACROFACET_SOURCE_DIR}/include/macrofacet/learned/CausalRenewalModel.h")
    message(FATAL_ERROR "Set TUNGSTEN_MACROFACET_SOURCE_DIR to the macrofacet checkout")
endif()
find_package(nanovdb CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
set(neural_sources
    mathutility/Gaussian1D.cpp mathutility/CubicPolynomial.cpp
    gpss/MeanField.cpp gpss/ShaderBallMean.cpp gpss/GPSSField.cpp gpss/CovarianceKernel.cpp
    gpss/RayMeanProfile.cpp gpss/Matern32Reference.cpp
    learned/RenewalHazardModel.cpp learned/CausalRenewalModel.cpp
    transport/RenewalRayDistribution.cpp transport/RenewalMedium.cpp
    fields/MeanFactory.cpp fields/NanoVdbMean.cpp fields/NanoVdbSampledField.cpp
    fieldgen/NanoVdbIO.cpp)
list(TRANSFORM neural_sources PREPEND "${TUNGSTEN_MACROFACET_SOURCE_DIR}/src/")
add_library(tungsten_neural_renewal STATIC ${neural_sources})
target_include_directories(tungsten_neural_renewal PRIVATE "${TUNGSTEN_MACROFACET_SOURCE_DIR}/include"
    PRIVATE "${TUNGSTEN_MACROFACET_SOURCE_DIR}/src")
target_link_libraries(tungsten_neural_renewal PUBLIC Eigen3::Eigen nlohmann_json::nlohmann_json
    PRIVATE nanovdb::nanovdb)
target_compile_features(tungsten_neural_renewal PUBLIC cxx_std_17)
set_property(SOURCE src/core/media/NeuralRenewalMedium.cpp src/core/media/MediumFactory.cpp
    src/core/integrators/TraceBase.cpp APPEND PROPERTY COMPILE_DEFINITIONS TUNGSTEN_ENABLE_NEURAL_RENEWAL=1)
set_property(SOURCE src/core/media/NeuralRenewalMedium.cpp APPEND PROPERTY
    INCLUDE_DIRECTORIES "${TUNGSTEN_MACROFACET_SOURCE_DIR}/include")
target_link_libraries(core PRIVATE "$<LINK_ONLY:tungsten_neural_renewal>")

option(TUNGSTEN_BUILD_NEURAL_RENEWAL_TESTS "Test the neural medium with an existing exported model" OFF)
if (TUNGSTEN_BUILD_NEURAL_RENEWAL_TESTS)
    enable_testing()
    add_executable(neural_renewal_test tests/NeuralRenewalTest.cpp)
    target_include_directories(neural_renewal_test PRIVATE "${TUNGSTEN_MACROFACET_SOURCE_DIR}/include")
    target_link_libraries(neural_renewal_test PRIVATE ${core_libs} tungsten_neural_renewal)
    add_test(NAME neural_renewal COMMAND neural_renewal_test
        "${TUNGSTEN_MACROFACET_SOURCE_DIR}/outputs/models/renewal_shader_ball_sigma002_r05_joint/renewal_model.json"
        "${TUNGSTEN_MACROFACET_SOURCE_DIR}/outputs/fields/bunny-64-sigma005.nvdb")
    foreach(mode furnace furnace_no_nee furnace_wavefront)
        add_test(NAME neural_renewal_${mode} COMMAND ${CMAKE_COMMAND}
            "-DRENDERER=$<TARGET_FILE:tungsten>" "-DCHECKER=$<TARGET_FILE:neural_renewal_test>"
            "-DSCENE=${CMAKE_CURRENT_SOURCE_DIR}/data/example-scenes/neural-renewal/bunny.json"
            "-DMACROFACET_DIR=${TUNGSTEN_MACROFACET_SOURCE_DIR}"
            "-DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/neural-renewal-${mode}-test"
            "-DMODE=${mode}" -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/NeuralRenewalRenderSmoke.cmake")
    endforeach()
endif()
