cmake_minimum_required(VERSION 3.20)

project(synth)

add_executable(main
  src/main.cpp
  src/deps/FastNoiseLite.h
  src/deps/stb_image_write.h
  src/deps/stb_image_write.c)

target_include_directories(main
  PUBLIC
    src/deps)

target_compile_features(main PUBLIC cxx_std_20)
