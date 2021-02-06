cmake_minimum_required(VERSION 3.8)
project(FingerprintAnalysis)

find_package(Doxygen REQUIRED)
#This will be the main output of our command
set(DOXYGEN_INDEX_FILE ${CMAKE_CURRENT_SOURCE_DIR}/html/index.html)
set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/docs/doxygen)
add_custom_command(OUTPUT ${DOXYGEN_INDEX_FILE}
        COMMAND ${DOXYGEN_EXECUTABLE} Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        MAIN_DEPENDENCY Doxyfile.doxygen
        COMMENT "Generating docs")

add_custom_target(Doxygen ALL DEPENDS ${DOXYGEN_INDEX_FILE})

find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( Prog main.cpp Image.cpp utils.cpp Pressure.cpp SymmetryTransformations.cpp MotionModel.cpp Filter.cpp Register.cpp)
target_link_libraries( Prog ${OpenCV_LIBS} )
