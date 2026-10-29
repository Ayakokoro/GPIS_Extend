cmake_minimum_required(VERSION 3.20)
project(Tungsten)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS Off)

option(TUNGSTEN_ENABLE_CUDA "Build the optional GPIS CUDA backend" OFF)
option(TUNGSTEN_ENABLE_NEURAL_RENEWAL "Build native neural GPIS rendering" ON)
option(TUNGSTEN_ENABLE_CONDITIONAL_PROFILE "Compile detailed conditional GP transport diagnostics" OFF)

# Disable annoying boost warning
set(Boost_NO_WARN_NEW_VERSIONS 1)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# AVX does not do much benefit at the moment, but breaks compilation on some platforms.
# Disabled for now until AVX becomes important enough to reconsider.
SET(EMBREE_MAX_ISA "SSE4.2" CACHE STRING "Selects highest ISA to support.")
set(USE_AVX FALSE CACHE BOOL "Use AVX.")
set(USE_AVX2 FALSE CACHE BOOL "Use AVX2.")
set(USE_SSE4_2 TRUE CACHE BOOL "Use SSE 4.2.")

include(OptimizeForArchitecture)
OptimizeForArchitecture()

if (MSVC)
    # Needed by MSVC, but not added by OptimizeForArchitexture()
    add_definitions(-D__SSE__)
endif()

add_definitions(-DINSTALL_PREFIX="${CMAKE_INSTALL_PREFIX}")

if (USE_AVX)
    message(STATUS "Compiling with AVX support")
    set(__AVX__ 1)
elseif (USE_SSE4_2)
    message(STATUS "Compiling with SSE4.2 support")
elseif (USE_SSSE3)
    message(STATUS "Compiling with SSE3 support")
else()
    message(FATAL_ERROR "The target machine does not support SSE3. At least SSE3 is required")
endif()

if (MSVC)
    # _USE_MATH_DEFINES makes MSVC's <cmath> expose M_PI and friends, which are a
    # POSIX extension rather than standard C++. It has to be a command line define
    # so that it is set before any header includes <cmath>. (NOMINMAX/CRT defines
    # stay here too — add_definitions applies to the whole directory.)
    add_definitions(-DCONSTEXPR=const -DNOMINMAX -D_CRT_SECURE_NO_WARNINGS -D_USE_MATH_DEFINES)
else()
    add_definitions(-DCONSTEXPR=constexpr)
endif()

if (MSVC)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/bigobj>)
endif ()

IF(COMMAND cmake_policy)
  if (POLICY CMP0043)
    cmake_policy(SET CMP0043 NEW)
  endif()
ENDIF(COMMAND cmake_policy)

foreach(flag ${Vc_ARCHITECTURE_FLAGS})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flag}")
endforeach()

set(EMBREE_STATIC_LIB ON CACHE BOOL "Build Embree as a static library." FORCE)
set(EMBREE_ISPC_SUPPORT OFF CACHE BOOL "Build Embree with support for ISPC applications." FORCE)
set(EMBREE_TUTORIALS OFF CACHE BOOL "Enable to build Embree tutorials" FORCE)
set(EMBREE_STAT_COUNTERS OFF CACHE BOOL "Enables statistic counters." FORCE)
set(EMBREE_RAY_MASK OFF CACHE BOOL "Enables ray mask support." FORCE)
set(EMBREE_BACKFACE_CULLING OFF CACHE BOOL "Enables backface culling." FORCE)
set(EMBREE_INTERSECTION_FILTER ON CACHE BOOL "Enables intersection filter callback." FORCE)
set(EMBREE_INTERSECTION_FILTER_RESTORE ON CACHE BOOL "Restores previous hit when hit is filtered out." FORCE)
set(EMBREE_TASKING_SYSTEM "INTERNAL" CACHE STRING "Selects tasking system" FORCE)
set(EMBREE_STATIC_RUNTIME OFF CACHE BOOL "Use the static version of the C/C++ runtime library." FORCE)
add_subdirectory(src/thirdparty/embree)
add_definitions(-DEMBREE_STATIC_LIB=1)

add_definitions(-DRAPIDJSON_HAS_STDSTRING=1)
add_definitions(-DSTBI_NO_STDIO=1)
add_definitions(-DLODEPNG_NO_COMPILE_DISK=1)
add_definitions(-DUSE_IPV6=1)
add_library(thirdparty STATIC
		src/thirdparty/civetweb/civetweb.c
		src/thirdparty/lodepng/lodepng.cpp
		src/thirdparty/sobol/sobol.cpp
		src/thirdparty/stbi/stb_image.c
		src/thirdparty/miniz/miniz.c
		src/thirdparty/skylight/ArHosekSkyModel.cpp
		src/thirdparty/tribox/tribox.cpp)

add_subdirectory(src/thirdparty/eigen)

set(AUTODIFF_BUILD_TESTS OFF)
set(AUTODIFF_BUILD_PYTHON OFF)
set(AUTODIFF_BUILD_EXAMPLES OFF)
set(AUTODIFF_BUILD_DOCS OFF)
add_subdirectory(src/thirdparty/autodiff)

add_subdirectory(src/thirdparty/spectra)


add_subdirectory(src/thirdparty/libigl)


if (CMAKE_COMPILER_IS_GNUCXX)
    set(CXX_WARNINGS "-Wall -Wextra -Wpointer-arith -Wcast-align -fstrict-aliasing -Wno-unused-local-typedefs -Wno-misleading-indentation -Wno-maybe-uninitialized -Wno-int-in-bool-context -Wno-implicit-fallthrough -Wno-class-memaccess -Wno-sign-compare -Wno-unknown-pragmas -Wno-unused-parameter -Wno-reorder -Wno-parentheses -Wno-pessimizing-move")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXX_WARNINGS} -fvisibility-inlines-hidden")
endif()

find_package(Boost REQUIRED)

set(core_libs core thirdparty embree Eigen3::Eigen autodiff::autodiff Spectra igl::core Boost::headers)

include_directories(
	src/core src/thirdparty 
	src/thirdparty/embree/include src 
	src/thirdparty/eigen/
	src/thirdparty/autodiff
	src/thirdparty/spectra/include
	src/thirdparty/libigl/include
	src/thirdparty/facet-forge/include
	${FCPW_EIGEN_INCLUDES})

find_package(Imath CONFIG REQUIRED)
find_package(OpenEXR CONFIG REQUIRED)
if (OpenEXR_FOUND AND Imath_FOUND)
    message(STATUS "OpenEXR detected. Building with .exr support")
    add_definitions(-DOPENEXR_AVAILABLE -DIMATH_HALF_NO_LOOKUP_TABLE)
    include_directories("/usr/local/include/Imath")
    set(core_libs ${core_libs} OpenEXR::OpenEXR Imath::Imath Imath::ImathConfig)
else()
    message(STATUS "No OpenEXR detected. Building without .exr support")
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
	set(core_libs ${core_libs} OpenMP::OpenMP_CXX)
    message("OPENMP FOUND")
else()
    message("Missing OPENMP!")
endif()

find_package(JPEG)
if (JPEG_FOUND)
    message(STATUS "JPEG library detected.")
    add_definitions(-DJPEG_AVAILABLE)
    include_directories(${JPEG_INCLUDE_DIR})
    set(core_libs ${core_libs} ${JPEG_LIBRARIES})
else()
    message(STATUS "No JPEG library detected. Falling back to stb_image JPG decoder")
endif()

find_package(TBB REQUIRED)
find_package(OpenVDB REQUIRED)
if (OpenVDB_FOUND AND OpenEXR_FOUND AND TBB_FOUND)
    message(STATUS "OpenVDB detected. Building with .vdb support")
    add_definitions(-DOPENVDB_AVAILABLE)
    add_definitions(-DOPENVDB_3_ABI_COMPATIBLE)
	add_definitions(-DQT_NO_KEYWORDS)
    include_directories(${OpenVDB_INCLUDE_DIRS} ${Tbb_INCLUDE_DIRS})
    set(core_libs ${core_libs} OpenVDB::openvdb TBB::tbb TBB::tbbmalloc)
	
#	if (NOT WIN32)
#		set(core_libs ${core_libs}
#			"/dartfs-hpc/rc/home/y/f003hxy/dev/vcpkg/installed/x64-linux/lib/liblz4.a"
#			"/dartfs-hpc/rc/home/y/f003hxy/dev/vcpkg/installed/x64-linux/lib/libzstd.a")
#	endif()

else()
    if (NOT OpenVDB_FOUND)
        message(STATUS "No OpenVDB detected. Building without .vdb support")
    elseif(NOT OpenEXR_FOUND)
        message(STATUS "No OpenEXR libraries detected. Building without .vdb support")
    else()
        message(STATUS "No TBB detected. Building without .vdb support")
    endif()
endif()

find_package(FFTW3 CONFIG)

if(FFTW3_FOUND)
    message(STATUS "FFTW detected. Building with DFT realization generation support")
	include_directories(${FFTW3_INCLUDE_DIRS})
	set(core_libs ${core_libs} FFTW3::fftw3)
	add_definitions(-DFFTW_AVAILABLE)
endif()

find_package(Ceres CONFIG)

if(Ceres_FOUND)
    message(STATUS "Ceres detected. Building with covariance fitting support.")
	include_directories(${Ceres_INCLUDE_DIRS})
	set(core_libs ${core_libs} Ceres::ceres)
	add_definitions(-DCERES_AVAILABLE)
endif()

file(GLOB_RECURSE Core_SOURCES "src/core/*.cpp")
if (NOT TUNGSTEN_ENABLE_NEURAL_RENEWAL)
    list(FILTER Core_SOURCES EXCLUDE REGEX "/math/neural/|/media/NeuralSdfGrid\\.cpp$")
endif()
add_library(core STATIC ${Core_SOURCES})
if (TUNGSTEN_ENABLE_CONDITIONAL_PROFILE)
    target_compile_definitions(core PUBLIC TUNGSTEN_ENABLE_CONDITIONAL_PROFILE=1)
endif()

# `core` is compiled before anything links the imported dependency targets, so it
# never inherited their include directories. OpenEXR's generated openexr_config.h
# does `#include <IlmThreadConfig.h>`, which only resolves when include/OpenEXR is
# on the path -- and that directory comes from OpenEXR::OpenEXRConfig, reachable
# only through OpenEXR::OpenEXR's INTERFACE_LINK_LIBRARIES, not from a plain
# property query. Linking the target is what actually pulls it in.
if (OpenEXR_FOUND AND Imath_FOUND)
    target_link_libraries(core PRIVATE OpenEXR::OpenEXR Imath::Imath)
endif()

add_library(facet-forge STATIC src/thirdparty/facet-forge/src/random.cpp)
set(core_libs ${core_libs} facet-forge)

if (OpenVDB_FOUND AND OpenEXR_FOUND AND TBB_FOUND)
    add_executable(bake_sdf_mean tools/sdf-baker/ProceduralMeanBaker.cpp)
    target_link_libraries(bake_sdf_mean PRIVATE ${core_libs})
endif()

option(TUNGSTEN_BUILD_CLASSIC_GLOBAL_TESTS "Build SDF-only classic_global tests" OFF)
option(TUNGSTEN_BUILD_GLOBAL_CONDITIONAL_TESTS "Build SDF-only conditional GP transport tests" OFF)
if (TUNGSTEN_BUILD_GLOBAL_CONDITIONAL_TESTS)
    enable_testing()
    add_executable(global_conditional_test tests/GlobalConditionalTest.cpp)
    target_link_libraries(global_conditional_test PRIVATE ${core_libs})
    # Isolate the private SDF arithmetic so tests can compare point queries to
    # the existing full Jet without exporting a reference-only production API.
    add_executable(sdf_point_jet_test tests/SdfPointJetTest.cpp src/core/math/Mat4f.cpp)
    target_link_libraries(sdf_point_jet_test PRIVATE Eigen3::Eigen)
    add_test(NAME global_conditional_point_jet COMMAND sdf_point_jet_test)
    add_test(NAME global_conditional_stage1 COMMAND global_conditional_test stage1)
    add_test(NAME global_conditional_stage2 COMMAND global_conditional_test stage2)
    add_test(NAME global_conditional_endpoint_precision COMMAND global_conditional_test endpoint_precision)
    add_test(NAME global_conditional_stage3 COMMAND global_conditional_test stage3)
    add_test(NAME global_conditional_surface_births COMMAND global_conditional_test surface_births)
    add_test(NAME global_conditional_sdf COMMAND global_conditional_test sdf)
    add_test(NAME global_conditional_knob COMMAND global_conditional_test knob)
    add_test(NAME global_conditional_bounds COMMAND global_conditional_test bounds)
    add_test(NAME global_conditional_tracking COMMAND global_conditional_test tracking)
    foreach(mode stage2 stage3 wavefront furnace furnace_no_nee knob)
        set(scene_name scene)
        set(render_mode ${mode})
        set(no_nee OFF)
        if (mode STREQUAL "stage2")
            set(scene_name stage2)
        elseif (mode STREQUAL "knob")
            set(scene_name knob)
        elseif (mode STREQUAL "furnace_no_nee")
            set(render_mode furnace)
            set(no_nee ON)
        endif()
        add_test(NAME global_conditional_${mode}_render COMMAND ${CMAKE_COMMAND}
            "-DRENDERER=$<TARGET_FILE:tungsten>"
            "-DCHECKER=$<TARGET_FILE:global_conditional_test>"
            "-DSCENE=${CMAKE_CURRENT_SOURCE_DIR}/data/example-scenes/global-conditional/${scene_name}.json"
            "-DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/global-conditional-${mode}-test"
            "-DMODE=${render_mode}" "-DNO_NEE=${no_nee}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/GlobalConditionalRenderSmoke.cmake")
    endforeach()
endif()
if (TUNGSTEN_BUILD_CLASSIC_GLOBAL_TESTS)
    enable_testing()
    add_executable(classic_global_test tests/ClassicGlobalTest.cpp)
    target_link_libraries(classic_global_test PRIVATE ${core_libs})
    add_test(NAME classic_global COMMAND classic_global_test "${CMAKE_CURRENT_BINARY_DIR}/classic-global-test")
    add_executable(classic_global_nonstationary_test tests/ClassicGlobalNonstationaryTest.cpp)
    target_link_libraries(classic_global_nonstationary_test PRIVATE ${core_libs})
    add_test(NAME classic_global_nonstationary COMMAND classic_global_nonstationary_test)
    add_test(NAME classic_global_nonstationary_render COMMAND ${CMAKE_COMMAND}
        "-DRENDERER=$<TARGET_FILE:tungsten>"
        "-DCHECKER=$<TARGET_FILE:classic_global_nonstationary_test>"
        "-DSCENE=${CMAKE_CURRENT_SOURCE_DIR}/data/example-scenes/classic-global/shaderball-nonstationary.json"
        "-DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/classic-global-nonstationary-render-test"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/ClassicGlobalNonstationaryRenderSmoke.cmake")
    add_test(NAME classic_global_render COMMAND ${CMAKE_COMMAND}
        "-DRENDERER=$<TARGET_FILE:tungsten>"
        "-DSCENE=${CMAKE_CURRENT_SOURCE_DIR}/data/example-scenes/classic-global/scene.json"
        "-DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/classic-global-render-test"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/ClassicGlobalRenderSmoke.cmake")
endif()

if (TUNGSTEN_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    enable_testing()
    add_subdirectory(src/gpu)
endif()

add_executable(obj2json src/obj2json/obj2json.cpp)
target_link_libraries(obj2json ${core_libs})

find_package(nanovdb CONFIG QUIET)
if (nanovdb_FOUND)
    target_link_libraries(core PRIVATE nanovdb::nanovdb)
    set_source_files_properties(src/core/media/ClassicLocalGrid.cpp PROPERTIES
        COMPILE_DEFINITIONS TUNGSTEN_NANOVDB_AVAILABLE=1)
endif()
# Analytic contact fixtures use this checkout's math implementation and write
# matching full-domain NanoVDB / OpenVDB fields for all four medium branches.
if (OpenVDB_FOUND AND TBB_FOUND AND nanovdb_FOUND)
    find_package(nlohmann_json CONFIG REQUIRED)
    add_executable(bake_contact_fields EXCLUDE_FROM_ALL
        tools/contact_experiments/bake.cpp
        src/core/math/conditional/mathutility/Gaussian1D.cpp)
    target_compile_definitions(bake_contact_fields PRIVATE NANOVDB_USE_OPENVDB)
    target_link_libraries(bake_contact_fields PRIVATE
        nanovdb::nanovdb OpenVDB::openvdb nlohmann_json::nlohmann_json Eigen3::Eigen)
    if (MSVC)
        target_compile_options(bake_contact_fields PRIVATE /EHsc /utf-8 /bigobj)
    endif()
    add_executable(surface_probe EXCLUDE_FROM_ALL tools/surface_compare/probe.cpp)
    target_link_libraries(surface_probe PRIVATE ${core_libs} nlohmann_json::nlohmann_json)
    add_executable(surface_path_trace EXCLUDE_FROM_ALL tools/surface_compare/path_trace.cpp)
    target_link_libraries(surface_path_trace PRIVATE ${core_libs} nlohmann_json::nlohmann_json)
    add_executable(surface_bounce_images EXCLUDE_FROM_ALL tools/surface_compare/bounce_images.cpp)
    target_link_libraries(surface_bounce_images PRIVATE ${core_libs} nlohmann_json::nlohmann_json)
    add_executable(gpis_shaderball_bake EXCLUDE_FROM_ALL
        tools/surface_compare/knob_bake.cpp src/core/math/SdfFunctionsQueries.cpp
        src/core/math/Mat4f.cpp src/core/math/conditional/mathutility/Gaussian1D.cpp)
    target_compile_definitions(gpis_shaderball_bake PRIVATE NANOVDB_USE_OPENVDB)
    target_link_libraries(gpis_shaderball_bake PRIVATE
        nanovdb::nanovdb OpenVDB::openvdb nlohmann_json::nlohmann_json Eigen3::Eigen)
    if (MSVC)
        target_compile_options(gpis_shaderball_bake PRIVATE /EHsc /utf-8 /bigobj)
    endif()
endif()

option(TUNGSTEN_BUILD_CLASSIC_LOCAL_TESTS "Build stationary classic_local tests" OFF)
if (TUNGSTEN_ENABLE_NEURAL_RENEWAL)
    include(cmake/NeuralRenewal.cmake)
endif()
if (TUNGSTEN_BUILD_CLASSIC_LOCAL_TESTS)
    if (NOT nanovdb_FOUND)
        message(FATAL_ERROR "classic_local tests require NanoVDB")
    endif()
    enable_testing()
    add_executable(classic_local_test tests/ClassicLocalTest.cpp)
    target_link_libraries(classic_local_test PRIVATE ${core_libs} nanovdb::nanovdb)
    add_test(NAME classic_local COMMAND classic_local_test "${CMAKE_CURRENT_BINARY_DIR}/classic-local-test")
    set_tests_properties(classic_local PROPERTIES FIXTURES_SETUP classic_local_field)
    foreach(local_mode nee no_nee wavefront boundary)
        add_test(NAME classic_local_${local_mode}_render COMMAND ${CMAKE_COMMAND}
            "-DRENDERER=$<TARGET_FILE:tungsten>" "-DCHECKER=$<TARGET_FILE:classic_local_test>"
            "-DSCENE=${CMAKE_CURRENT_BINARY_DIR}/classic-local-test/scene.json"
            "-DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/classic-local-${local_mode}-render"
            "-DMODE=${local_mode}" -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/ClassicLocalRenderSmoke.cmake")
        set_tests_properties(classic_local_${local_mode}_render PROPERTIES FIXTURES_REQUIRED classic_local_field)
    endforeach()
endif()
if (OpenVDB_FOUND AND OpenEXR_FOUND AND TBB_FOUND AND nanovdb_FOUND)
    add_executable(obj2json_macrofacet src/obj2json/obj2json_macrofacet.cpp)
    target_link_libraries(obj2json_macrofacet PRIVATE ${core_libs} nanovdb::nanovdb)
endif()

add_executable(json2xml src/json2xml/json2xml.cpp)
target_link_libraries(json2xml ${core_libs})

add_executable(scenemanip src/scenemanip/scenemanip.cpp)
target_link_libraries(scenemanip ${core_libs})

add_executable(hdrmanip src/hdrmanip/hdrmanip.cpp)
target_link_libraries(hdrmanip ${core_libs})

if (EIGEN3_FOUND)
    file(GLOB_RECURSE denoiser_SOURCES "src/denoiser/*.cpp")
    add_executable(denoiser ${denoiser_SOURCES})
    target_link_libraries(denoiser ${core_libs})
endif()

add_executable(tungsten src/tungsten/tungsten.cpp)
target_link_libraries(tungsten ${core_libs})
if (TARGET tungsten_gp_cuda_pathtracer)
    target_compile_definitions(tungsten PRIVATE TUNGSTEN_ENABLE_CUDA=1)
    target_link_libraries(tungsten tungsten_gp_cuda_pathtracer)
endif()

if (WIN32)
    set(socket_libs wsock32 Ws2_32)
else()
    set(socket_libs "")
endif()

add_executable(tungsten_server src/tungsten-server/tungsten-server.cpp)
target_link_libraries(tungsten_server ${core_libs} ${socket_libs})

set(executables obj2json json2xml scenemanip hdrmanip tungsten tungsten_server)
if (TARGET obj2json_macrofacet)
    list(APPEND executables obj2json_macrofacet)
endif()
if (EIGEN3_FOUND)
    set(executables ${executables} denoiser)
endif()
set(data_dirs example-scenes materialtest mc-loader)


find_package(OpenGL)
option(BUILD_EDITOR "Build the editor" FALSE)
# Qt5 is only needed by the editor. Looking it up unconditionally made Qt5 a hard
# dependency of every build, even BUILD_EDITOR=OFF ones.
if (BUILD_EDITOR)
    find_package(Qt5 COMPONENTS Widgets OpenGL REQUIRED)
endif()
if (BUILD_EDITOR AND OPENGL_FOUND AND Qt5Widgets_FOUND)
	set(CMAKE_AUTOMOC ON)
	set(CMAKE_AUTORCC ON)
	set(CMAKE_AUTOUIC ON)
	set(CMAKE_INCLUDE_CURRENT_DIR ON)

    include_directories(${OPENGL_INCLUDE_DIR})
    set(opengl_libs ${OPENGL_LIBRARIES})
	
    file(GLOB_RECURSE Editor_SOURCES "src/editor/*.*pp")
    if (APPLE)
        list(APPEND Editor_SOURCES "src/editor/CoreProfileAttributes.mm")
    endif()
    add_executable(editor ${Editor_SOURCES} src/editor/resources/Tungsten.rc)


    set_target_properties(editor PROPERTIES AUTOMOC TRUE)
    target_link_libraries(editor ${core_libs} ${opengl_libs} Qt5::Core Qt5::Widgets Qt5::OpenGL)
    if (APPLE)
        find_library(APPKIT NAMES AppKit) 
        target_link_libraries(editor ${APPKIT})
    endif()
    message(STATUS "Building editor")
    
    set(data_dirs ${data_dirs} editor shaders)
    set(executables ${executables} editor)
else()
    if (BUILD_EDITOR AND NOT Qt5Widgets_FOUND)
        message(STATUS "Qt5 not found. Editor will not be built")
    endif()
    if (NOT OPENGL_FOUND)
        message(STATUS "OpenGL not found. Editor will not be built")
    endif()
	if (NOT BUILD_EDITOR)
        message(STATUS "Editor build disabled. Editor will not be built")
    endif()
endif()

if (WIN32)
    set(data_prefix "data")
else()
    set(data_prefix "share/tungsten")
endif()

#foreach (data_dir ${data_dirs})
#    install(DIRECTORY data/${data_dir} DESTINATION ${data_prefix})
#    file(GLOB_RECURSE data_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/data "data/${data_dir}/*")
#    foreach (data ${data_files})
#        configure_file(data/${data} ${CMAKE_CURRENT_BINARY_DIR}/${data_prefix}/${data} COPYONLY)
#    endforeach(data)
#endforeach()

if (WIN32)
    install(TARGETS ${executables} DESTINATION .)
else()
    install(TARGETS ${executables} DESTINATION bin)
endif()
