file(MAKE_DIRECTORY "${OUTPUT_DIR}")
file(READ "${SCENE}" scene)
string(JSON scene SET "${scene}" camera resolution "[64,64]")
file(WRITE "${OUTPUT_DIR}/scene.json" "${scene}")
execute_process(
    COMMAND "${RENDERER}" -t 2 --spp 2 -d "${OUTPUT_DIR}" "${OUTPUT_DIR}/scene.json"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE errors TIMEOUT 180
)
file(WRITE "${OUTPUT_DIR}/render.log" "${output}\n${errors}")
if (NOT "${result}" STREQUAL "0" OR
    "${output}\n${errors}" MATCHES "Caught an internal error|unrecoverable error|GPContext does not exist|Unable to load|not implemented|Wrong normal direction|terminate called")
    message(FATAL_ERROR "nonstationary shaderball render failed (${result}):\n${output}\n${errors}")
endif()
if (NOT EXISTS "${OUTPUT_DIR}/shaderball-nonstationary-classic-global.png")
    message(FATAL_ERROR "nonstationary shaderball render did not produce its PNG")
endif()
execute_process(
    COMMAND "${CHECKER}" "${OUTPUT_DIR}/shaderball-nonstationary-classic-global.exr"
    RESULT_VARIABLE result OUTPUT_VARIABLE stats ERROR_VARIABLE errors
)
file(WRITE "${OUTPUT_DIR}/validation.log" "${stats}\n${errors}")
if (NOT "${result}" STREQUAL "0")
    message(FATAL_ERROR "nonstationary shaderball image validation failed:\n${stats}\n${errors}")
endif()
message(STATUS "${stats}")
