file(MAKE_DIRECTORY "${OUTPUT_DIR}")
execute_process(
    COMMAND "${RENDERER}" -t 2 --spp 1 -d "${OUTPUT_DIR}" "${SCENE}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE errors
    TIMEOUT 90
)
file(WRITE "${OUTPUT_DIR}/render.log" "${output}\n${errors}")
if (NOT "${result}" STREQUAL "0" OR
    "${output}\n${errors}" MATCHES "Caught an internal error|GPContext does not exist|Unable to load|not implemented|Wrong normal direction")
    message(FATAL_ERROR "classic_global mixed-scene render failed (${result}):\n${output}\n${errors}")
endif()
if (NOT EXISTS "${OUTPUT_DIR}/classic-global.png" OR NOT EXISTS "${OUTPUT_DIR}/classic-global.exr")
    message(FATAL_ERROR "classic_global mixed-scene render did not produce its images")
endif()
