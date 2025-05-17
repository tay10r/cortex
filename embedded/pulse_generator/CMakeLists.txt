cmake_minimum_required(VERSION 3.20)

project(pulse_generator_tests)

find_package(GTest CONFIG REQUIRED)

add_executable(pg-tests
  program.h
  program.cpp
  tests/program.cpp
)

target_link_libraries(pg-tests
  PUBLIC
    GTest::gtest
    GTest::gtest_main
)
