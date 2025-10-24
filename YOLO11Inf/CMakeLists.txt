cmake_minimum_required(VERSION 3.8)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project("YOLO11Inf")

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Platform-specific configuration
if(WIN32)
    # Windows: Set library paths
    set(OpenCV_DIR "C:/tools/opencv/build/x64/vc16/lib" CACHE PATH "Path to OpenCV config")
    set(ONNXRUNTIME_ROOT_PATH "C:/onnxruntime/onnxruntime-win-x64-1.23.1" CACHE PATH "Path to ONNX Runtime")
endif()

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find ONNX Runtime
if(WIN32)
    # Windows: Use manually specified path
    set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_ROOT_PATH}/include")
    set(ONNXRUNTIME_LIB_DIRS "${ONNXRUNTIME_ROOT_PATH}/lib")
    
    find_library(ONNXRUNTIME_LIBRARY
        NAMES onnxruntime
        PATHS ${ONNXRUNTIME_LIB_DIRS}
        NO_DEFAULT_PATH
    )
else()
    # Linux: Try to find via pkg-config or standard paths
    find_path(ONNXRUNTIME_INCLUDE_DIRS
        NAMES onnxruntime_cxx_api.h
        PATHS /usr/include /usr/local/include
        PATH_SUFFIXES onnxruntime onnxruntime/core/session
    )
    
    find_library(ONNXRUNTIME_LIBRARY
        NAMES onnxruntime
        PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu
    )
endif()

# Check if ONNX Runtime was found
if(NOT ONNXRUNTIME_LIBRARY OR NOT ONNXRUNTIME_INCLUDE_DIRS)
    message(FATAL_ERROR "ONNX Runtime not found. Please set ONNXRUNTIME_ROOT_PATH on Windows or install ONNX Runtime on Linux")
endif()

message(STATUS "ONNX Runtime Include: ${ONNXRUNTIME_INCLUDE_DIRS}")
message(STATUS "ONNX Runtime Library: ${ONNXRUNTIME_LIBRARY}")
message(STATUS "OpenCV Version: ${OpenCV_VERSION}")
message(STATUS "OpenCV Libraries: ${OpenCV_LIBS}")

# Add include directories
include_directories(${ONNXRUNTIME_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

# Add source to this project's executable
add_executable(YOLO11Inf "img_inference.cpp" "YOLO11Det.hpp")

# Link libraries
target_link_libraries(YOLO11Inf 
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIBRARY}
)

# Linux may need pthread
if(UNIX)
    target_link_libraries(YOLO11Inf pthread)
endif()

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET YOLO11Inf PROPERTY CXX_STANDARD 20)
endif()

# Copy ONNX Runtime DLL on Windows
if(WIN32)
    add_custom_command(TARGET YOLO11Inf POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_ROOT_PATH}/lib/onnxruntime.dll"
        $<TARGET_FILE_DIR:YOLO11Inf>
    )
endif()