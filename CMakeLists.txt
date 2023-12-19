cmake_minimum_required(VERSION 3.0.0)
project(CornerDetection VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DiRS})

add_executable(CornerDetection main.cpp)

target_link_libraries( CornerDetection ${OpenCV_LIBS})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
