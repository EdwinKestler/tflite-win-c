cmake_minimum_required(VERSION 3.10)

# Project Name
project(tflite-win-c)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# TensorFlow Lite
set(TFLITE_ROOT "E:/tflite-dist")
include_directories("${TFLITE_ROOT}/include")
link_directories("${TFLITE_ROOT}/libs/windows_x86_64")

# OpenCV
set(OPENCV_ROOT "E:/opencv/build")
find_package(OpenCV REQUIRED PATHS "${OPENCV_ROOT}" NO_DEFAULT_PATH)
include_directories(${OpenCV_INCLUDE_DIRS})

# Application Source
add_executable(tflite-c-win Main.cpp  ObjectDetector.cpp rstpStream.cpp)

# Link Libraries
target_link_libraries(tflite-c-win ${OpenCV_LIBS} "E:/tflite-dist/libs/windows_x86_64/tensorflowlite_c.dll.if.lib")

# Copy necessary DLLs to output directory
add_custom_command(TARGET tflite-c-win POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${TFLITE_ROOT}/libs/windows_x86_64/tensorflowlite_c.dll"
    $<TARGET_FILE_DIR:tflite-c-win>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${OPENCV_ROOT}/x64/vc16/bin/opencv_world490.dll"
    $<TARGET_FILE_DIR:tflite-c-win>)