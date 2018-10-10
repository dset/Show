cmake_minimum_required(VERSION 3.12)
project(show)

find_package(OpenCV REQUIRED)
find_package(OpenCL REQUIRED)

set(CMAKE_CXX_STANDARD 14)

add_executable(show main.cpp bmp.cpp bmp.h image.cpp image.h math.cpp math.h ImageProcessor.cpp ImageProcessor.h OpenCLEnvironment.cpp OpenCLEnvironment.h)

target_link_libraries(show ${OpenCV_LIBS} ${OpenCL_LIBRARIES})