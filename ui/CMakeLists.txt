cmake_minimum_required(VERSION 3.18)

project(cortex_ui
  DESCRIPTION "The UI component of the web app."
  VERSION 1.0)

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

set(sources
  app.cpp
  src/widget.h
  src/camera_widget.h
  src/camera_widget.cpp
  src/sangaboard_widget.h
  src/sangaboard_widget.cpp
  src/gallery_widget.h
  src/gallery_widget.cpp
  src/visualizer.h
  src/visualizer.cpp
  src/transfer.h
  src/image_index.h
  src/image_index.cpp
  buffer.h
  decoder.h
  decoder.cpp
  viewport.h
  viewport.cpp
  config.h
  config.cpp
  tool.h
  tool.cpp
  tools/focus_tool.h
  tools/focus_tool.cpp
  ${texteditor_SOURCE_DIR}/TextEditor.h
  ${texteditor_SOURCE_DIR}/TextEditor.cpp
)

if(EMSCRIPTEN)
  list(APPEND sources src/transfer_real.cpp)
else()
  list(APPEND sources src/transfer_fake.cpp)
endif()

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
)

if(EMSCRIPTEN)
  #target_compile_options(cortex_ui PRIVATE "SHELL: -s FETCH=1")
  # initial memory = 48 MiB
  target_link_options(cortex_ui PRIVATE "SHELL: -sFETCH -sINITIAL_MEMORY=50331648")
endif()
