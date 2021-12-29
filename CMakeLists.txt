cmake_minimum_required(VERSION 3.2)

project(gstreamer_apriltag VERSION 1.0)

# -std=gnu++11
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-write-strings -O3 -Os")

configure_file(inc/projectConfig.h.in inc/projectConfig.h)

# setup CUDA
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/utils/cuda")
find_package(CUDA)
message("-- CUDA version: ${CUDA_VERSION}")

set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS}; 
    -O3 
    -gencode arch=compute_37,code=sm_37
    -gencode arch=compute_53,code=sm_53
    -gencode arch=compute_60,code=sm_60
    -gencode arch=compute_61,code=sm_61
    -gencode arch=compute_62,code=sm_62
)
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

if(CUDA_VERSION_MAJOR GREATER 9)
    message("-- CUDA ${CUDA_VERSION_MAJOR} detected, enabling SM_72")

    set(
        CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS}; 
        -gencode arch=compute_72,code=sm_72
    )
endif()

find_package(OpenCV REQUIRED)

if( NOT OpenCV_FOUND )
    message("-- didn't find OpenCV on system, disabling OpenCV")
else()
    message("-- OpenCV version:  " ${OpenCV_VERSION})

    if( ${OpenCV_VERSION_MAJOR} LESS 3 )
        message("-- OpenCV version less than 3.0, disabling OpenCV")
    else()
        message("-- OpenCV version >= 3.0.0, enabling OpenCV")
        set(HAS_OPENCV 1)
        add_definitions(-DHAS_OPENCV)
    endif()	
endif()

# VPI (workaround for inherited dependency from isaac_ros_image_proc)
find_package(vpi REQUIRED)

# Eigen
find_package(Eigen3 REQUIRED)
find_package(Threads REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})



set(PROJECT_OUTPUT_DIR  ${PROJECT_BINARY_DIR}/${CMAKE_SYSTEM_PROCESSOR})
set(PROJECT_INCLUDE_DIR ${PROJECT_OUTPUT_DIR}/include)

file(MAKE_DIRECTORY ${PROJECT_INCLUDE_DIR})
file(MAKE_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin)

message("-- system arch:  ${CMAKE_SYSTEM_PROCESSOR}")
message("-- output path:  ${PROJECT_OUTPUT_DIR}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/lib)


add_subdirectory(utils)
	# build C/C++ interface
link_directories(/usr/lib/aarch64-linux-gnu/tegra)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/inc)
include_directories(${PROJECT_INCLUDE_DIR} ${PROJECT_INCLUDE_DIR}/jetson-inference ${PROJECT_INCLUDE_DIR}/jetson-utils)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/nvapriltags/nvapriltags)
include_directories(/usr/include/gstreamer-1.0 /usr/lib/aarch64-linux-gnu/gstreamer-1.0/include /usr/include/glib-2.0 /usr/include/libxml2 /usr/lib/aarch64-linux-gnu/glib-2.0/include/)
# include_directories( ${PROJECT_INCLUDE_DIR}/nvapriltags/nvapriltags ${CMAKE_CURRENT_SOURCE_DIR}/utils ${PROJECT_INCLUDE_DIR}/utils/*)
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/utils/camera)
# include_directories( ${PROJECT_INCLUDE_DIR}/nvapriltags/nvapriltags ${PROJECT_INCLUDE_DIR}/utils)



execute_process(COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE)
message( STATUS "Architecture: ${ARCHITECTURE}" )
message( STATUS "CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}" )

add_executable(gstreamer_apriltag ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)


add_library(nvapriltags STATIC IMPORTED)
if( ${ARCHITECTURE} STREQUAL "x86_64" )
  set_property(TARGET nvapriltags PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/nvapriltags/lib_x86_64/libapril_tagging.a)
elseif( ${ARCHITECTURE} STREQUAL "aarch64" )
  set_property(TARGET nvapriltags PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/nvapriltags/lib_aarch64_jetpack44/libapril_tagging.a)
endif()

target_link_libraries(gstreamer_apriltag PUBLIC nvapriltags ${CUDA_LIBRARIES} vpi jetson-utils ${OpenCV_LIBS})




