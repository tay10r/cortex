cmake_minimum_required(VERSION 3.20)

project(cortex)

if(DEFINED CMAKE_BUILD_TYPE)
  message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
endif()

find_package(OpenCV REQUIRED)
find_package(nlohmann_json REQUIRED)

if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

include(FetchContent)

FetchContent_Declare(pybind11
  URL "https://github.com/pybind/pybind11/archive/refs/tags/v2.13.6.zip"
  URL_HASH "SHA256=d0a116e91f64a4a2d8fb7590c34242df92258a61ec644b79127951e821b47be6")
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(cortex
  src/main.cpp
  src/api.h
  src/api.cpp
  src/wbc_differential_api.h
  src/wbc_differential_api.cpp
  src/wbc_classifier.h
  src/wbc_classifier.cpp
  src/wbc_localizer.h
  src/wbc_localizer.cpp)

target_link_libraries(cortex
  PUBLIC
    ${OpenCV_LIBRARIES}
    nlohmann_json::nlohmann_json)

install(TARGETS cortex
  RUNTIME DESTINATION "."
  LIBRARY DESTINATION "."
  ARCHIVE DESTINATION ".")
