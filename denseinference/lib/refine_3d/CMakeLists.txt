cmake_minimum_required (VERSION 3.3)
project(dense_inference)

MESSAGE(STATUS "Configuring ${PROJECT_NAME}...")

if(POLICY CMP0020)
  cmake_policy(SET CMP0020 NEW)
endif()

set(CMAKE_BUILD_TYPE Release)

#python
# SET(Python_ADDITIONAL_VERSIONS 3)
FIND_PACKAGE(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
get_filename_component(Python_Root ${PYTHON_EXECUTABLE} DIRECTORY)

#FindBoost
find_package(Boost COMPONENTS python REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

FILE(GLOB DenseInference_SRC *.cpp)
FILE(GLOB DenseInference_HDR *.h)

# numpy
SET(NUMPY_INCLUDE "${Python_Root}/Lib/site-packages/numpy/core/include" CACHE PATH "Path to numpy include folder")
if(NOT NUMPY_INCLUDE)
    message(FATAL_ERROR "numpy is not found")
endif()
include_directories(${NUMPY_INCLUDE})

include_directories(${CMAKE_SOURCE_DIR})


add_library(${PROJECT_NAME} SHARED ${DenseInference_SRC} ${DenseInference_HDR})
target_link_libraries(${PROJECT_NAME} ${PYTHON_LIBRARY} ${Boost_LIBRARIES} libDenseCRF)