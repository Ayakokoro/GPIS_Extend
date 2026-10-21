file(MAKE_DIRECTORY "${OUTPUT_DIR}")
file(READ "${SCENE}" scene)
get_filename_component(scene_dir "${SCENE}" DIRECTORY)
get_filename_component(geometry_dir "${scene_dir}/../classic-global" ABSOLUTE)
string(REPLACE "../classic-global/" "${geometry_dir}/" scene "${scene}")
set(spp 4)
if (MODE STREQUAL "wavefront")
    string(JSON scene SET "${scene}" integrator wavefront_gpis true)
endif()
if (MODE STREQUAL "furnace")
    set(spp 128)
    string(JSON scene SET "${scene}" camera resolution "[16,16]")
    string(JSON scene SET "${scene}" media 0 material albedo "[1,1,1]")
    string(JSON scene SET "${scene}" primitives 1 bsdf albedo 1)
    string(JSON scene SET "${scene}" primitives 2 bsdf albedo 1)
    # An ordinary forward triangle inside the medium must renew the observation.
    string(JSON scene SET "${scene}" primitives 4
        "{\"type\":\"mesh\",\"file\":\"${geometry_dir}/triangle.obj\",\"bsdf\":\"boundary\",\"transform\":{\"position\":[0,0.4,0]}}")
    if (NO_NEE)
        string(JSON scene SET "${scene}" integrator enable_light_sampling false)
        string(JSON scene SET "${scene}" integrator enable_volume_light_sampling false)
    endif()
endif()
file(WRITE "${OUTPUT_DIR}/scene.json" "${scene}")
execute_process(
    COMMAND "${RENDERER}" -t 2 --spp "${spp}" -d "${OUTPUT_DIR}" "${OUTPUT_DIR}/scene.json"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE errors TIMEOUT 180
)
file(WRITE "${OUTPUT_DIR}/render.log" "${output}\n${errors}")
if (NOT "${result}" STREQUAL "0" OR
    "${output}\n${errors}" MATCHES "Caught an internal error|unrecoverable error|GPContext does not exist|Unable to load|not implemented|Wrong normal direction|terminate called")
    message(FATAL_ERROR "global_conditional render failed (${result}):\n${output}\n${errors}")
endif()
set(check image)
if (MODE STREQUAL "furnace")
    set(check furnace)
endif()
execute_process(
    COMMAND "${CHECKER}" "${check}" "${OUTPUT_DIR}/global-conditional.exr"
    RESULT_VARIABLE result OUTPUT_VARIABLE stats ERROR_VARIABLE errors
)
file(WRITE "${OUTPUT_DIR}/validation.log" "${stats}\n${errors}")
if (NOT "${result}" STREQUAL "0")
    message(FATAL_ERROR "conditional render validation failed:\n${stats}\n${errors}")
endif()
message(STATUS "${stats}")
