# TestingSetup.cmake
# This file handles creating symbolic link to engine and image file for testing

# Include shared model configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake)

# Derived paths
set(ENGINE_LINK_PATH ${CMAKE_CURRENT_BINARY_DIR}/${ENGINE_FILE})

# Create symbolic link to engine file in build directory for testing
add_custom_command(
  OUTPUT ${ENGINE_LINK_PATH}
  COMMAND ${CMAKE_COMMAND} -E create_symlink
          ${ENGINE_PATH}
          ${ENGINE_LINK_PATH}
  DEPENDS rf_detr_trt_backend  # Depend on the main library target so it won't build before the engine is generated
  COMMENT "Creating symbolic link to engine file for testing: ${ENGINE_FILE}..."
)

# Custom target for the symbolic link
add_custom_target(test_engine_link
  DEPENDS ${ENGINE_LINK_PATH}
)

# Initialize an empty list to collect output files
set(TEST_IMAGE_OUTPUTS)

# Loop to create a symbolic link for each image file in build directory for testing
foreach(image_file IN LISTS TEST_IMAGE_FILES)
  set(src ${CMAKE_CURRENT_SOURCE_DIR}/test/${image_file})
  set(dst ${CMAKE_CURRENT_BINARY_DIR}/${image_file})

  add_custom_command(
    OUTPUT ${dst}
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${src} ${dst}
    DEPENDS rf_detr_trt_backend
    COMMENT "Creating symbolic link to ${image_file} for testing..."
  )

  list(APPEND TEST_IMAGE_OUTPUTS ${dst})
endforeach()

# Custom target that depends on all symbolic links
add_custom_target(test_image_link
  DEPENDS ${TEST_IMAGE_OUTPUTS}
)

# Function to add testing dependency to a target
function(add_testing_dependency target_name)
  add_dependencies(${target_name} test_engine_link test_image_link)
endfunction()
