cmake_minimum_required (VERSION 2.8.3)
project(libDenseCRF)

MESSAGE(STATUS "Configuring ${PROJECT_NAME}...")

if(POLICY CMP0020)
  cmake_policy(SET CMP0020 NEW)
endif()

set(CMAKE_BUILD_TYPE Release)

#python
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

FILE(GLOB LibDenseCRF_SRC *.cpp)
FILE(GLOB LibDenseCRF_HDR *.h)

add_library(${PROJECT_NAME} STATIC ${LibDenseCRF_SRC} ${LibDenseCRF_HDR})
target_link_libraries(${PROJECT_NAME})