cmake_minimum_required(VERSION 3.18)

project(cortex_ui
  DESCRIPTION "The UI component of the web app."
  VERSION 1.0
)

if(NOT EMSCRIPTEN)
  find_package(Boost CONFIG REQUIRED COMPONENTS system)
endif()

if(NOT DEFINED MEMORY_LIMIT)
  # This is the memory limit for the app as a whole,
  # which includes the stack size, heap size (including all untracked allocations).
  # There should be some memory for different types of small objects, and memory specifically for:
  #  - a download of a 3280x2464 16 bpp image from the camera
  #  - a download of a 3280x2464 16 bpp image from the gallery
  #  - a few working copies of an image
  # 128 MiB
  set(MEMORY_LIMIT 134217728)
endif()

if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

include(FetchContent)

set(UIKIT_MAIN ON CACHE BOOL "" FORCE)
FetchContent_Declare(uikit
  URL "https://github.com/tay10r/uikit/archive/refs/tags/v0.2.1.zip"
  URL_HASH "SHA256=ea83f5bc2c34365d19808cca1d1390cad42e60bccc8844092214edbc9d945dea")
FetchContent_MakeAvailable(uikit)

FetchContent_Declare(texteditor
  URL "https://github.com/BalazsJako/ImGuiColorTextEdit/archive/0a88824f7de8d0bd11d8419066caa7d3469395c4.zip"
  URL_HASH "SHA256=8d9c06a1cf0b46df8b5f7afe0cade0d9bbea717dbf4d80b029cc2660fcedb6fa")
FetchContent_MakeAvailable(texteditor)
FetchContent_GetProperties(texteditor)

add_library(stb
  stb/stb_image.h
  stb/stb_image_write.h
  stb/stb.c
)

target_include_directories(stb PUBLIC stb)

include(cmake/CMakeRC.cmake)

cmrc_add_resource_library(shaders
  shaders/debayer.vert
  shaders/debayer.frag
)

#=================#
# Support Library #
#=================#

add_library(cortex
  src/memory.h
  src/memory.cpp
  src/image.h
  src/image.cpp
  src/task.h
  src/white_balance.h
  src/white_balance.cpp
)

if(EMSCRIPTEN)
  target_sources(cortex PRIVATE src/task_browser.cpp)
else()
  target_sources(cortex PRIVATE src/task_desktop.cpp)
endif()

target_compile_definitions(cortex
  PUBLIC
    SOFT_MEMORY_LIMIT=${SOFT_MEMORY_LIMIT}
)

if(NOT EMSCRIPTEN)
  target_link_libraries(cortex PUBLIC Boost::system)
endif()

#==============#
# Main Program #
#==============#

set(sources
  app.cpp
  src/plot.h
  src/widget.h
  src/camera_widget.h
  src/camera_widget.cpp
  src/sangaboard_widget.h
  src/sangaboard_widget.cpp
  src/gallery_widget.h
  src/gallery_widget.cpp
  src/visualizer.h
  src/visualizer.cpp
  src/image_index.h
  src/image_index.cpp
  src/tools_widget.h
  src/tools_widget.cpp
  src/tool.h
  src/tool.cpp
  src/white_balance_tool.h
  src/white_balance_tool.cpp
  ${texteditor_SOURCE_DIR}/TextEditor.h
  ${texteditor_SOURCE_DIR}/TextEditor.cpp
)

add_executable(cortex_ui WIN32 ${sources})

target_include_directories(cortex_ui
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/deps/json"
    "${texteditor_SOURCE_DIR}"
)

target_link_libraries(cortex_ui
  PRIVATE
    uikit::uikit
    uikit::main
    stb
    shaders
    cortex
)

target_compile_features(cortex_ui PUBLIC cxx_std_17)

if(EMSCRIPTEN)
  #target_compile_options(cortex_ui PRIVATE "SHELL: -s FETCH=1")
  # initial memory = 48 MiB
  target_link_options(cortex_ui PRIVATE "SHELL: -sFETCH -sINITIAL_MEMORY=${MEMORY_LIMIT}")
endif()

# For local testing and deployment to Github pages:
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/../services/app/static/index.html"
  DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
