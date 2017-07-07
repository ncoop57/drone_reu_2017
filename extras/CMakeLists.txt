cmake_minimum_required (VERSION 2.8)

set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

project (gpu)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(gpu gpu.cpp)

target_link_libraries(gpu ${OpenCV_LIBS})

