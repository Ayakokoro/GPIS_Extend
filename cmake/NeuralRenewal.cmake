# Native inference, SDF input and flight sampling are compiled into core.
find_package(nanovdb CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
target_link_libraries(core PRIVATE nlohmann_json::nlohmann_json nanovdb::nanovdb)
set_property(SOURCE src/core/media/NeuralRenewalMedium.cpp src/core/media/MediumFactory.cpp
    src/core/integrators/TraceBase.cpp APPEND PROPERTY COMPILE_DEFINITIONS TUNGSTEN_ENABLE_NEURAL_RENEWAL=1)

option(TUNGSTEN_BUILD_NEURAL_RENEWAL_TESTS "Test the neural medium with an existing exported model" OFF)
if (TUNGSTEN_BUILD_NEURAL_RENEWAL_TESTS)
    set(TUNGSTEN_NEURAL_RENEWAL_TEST_MODEL
        "${PROJECT_SOURCE_DIR}/models/renewal_shader_ball_sigma002_r05_joint/renewal_model.json"
        CACHE FILEPATH "Exported joint model for neural renewal tests")
    set(TUNGSTEN_NEURAL_RENEWAL_TEST_FIELD
        "${PROJECT_SOURCE_DIR}/gpis-scenes/bunny/bunny-64-sigma005.nvdb"
        CACHE FILEPATH "Full-domain bunny SDF for neural renewal tests")
    foreach(asset TUNGSTEN_NEURAL_RENEWAL_TEST_MODEL TUNGSTEN_NEURAL_RENEWAL_TEST_FIELD)
        if (NOT EXISTS "${${asset}}" OR IS_DIRECTORY "${${asset}}")
            message(FATAL_ERROR
                "Neural renewal tests require ${asset}: ${${asset}}\n"
                "Set -D${asset}=<file>, or build without these runtime assets using "
                "-DTUNGSTEN_BUILD_NEURAL_RENEWAL_TESTS=OFF.")
        endif()
    endforeach()
    enable_testing()
    add_executable(neural_renewal_test tests/NeuralRenewalTest.cpp)
    target_link_libraries(neural_renewal_test PRIVATE ${core_libs} nlohmann_json::nlohmann_json)
    add_test(NAME neural_renewal COMMAND neural_renewal_test
        "${TUNGSTEN_NEURAL_RENEWAL_TEST_MODEL}"
        "${TUNGSTEN_NEURAL_RENEWAL_TEST_FIELD}")
    foreach(mode furnace furnace_no_nee furnace_wavefront)
        add_test(NAME neural_renewal_${mode} COMMAND ${CMAKE_COMMAND}
            "-DRENDERER=$<TARGET_FILE:tungsten>" "-DCHECKER=$<TARGET_FILE:neural_renewal_test>"
            "-DSCENE=${CMAKE_CURRENT_SOURCE_DIR}/tests/NeuralRenewalRenderScene.json.in"
            "-DNEURAL_MODEL=${TUNGSTEN_NEURAL_RENEWAL_TEST_MODEL}"
            "-DNEURAL_FIELD=${TUNGSTEN_NEURAL_RENEWAL_TEST_FIELD}"
            "-DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/neural-renewal-${mode}-test"
            "-DMODE=${mode}" -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/NeuralRenewalRenderSmoke.cmake")
    endforeach()
endif()
