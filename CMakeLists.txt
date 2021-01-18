cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
set (CMAKE_CXX_STANDARD 11)

project( Coins )

find_package( OpenCV REQUIRED )

include_directories(${OpenCV_INCLUDE_DIRS})

add_library(coins_toolbox
	coins_toolbox.hpp
	coins_toolbox.cpp
)

add_executable(${PROJECT_NAME} coins.cpp )
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )
