cmake_minimum_required(VERSION 3.5)
project(SudokuSolver)

set("OpenCV_DIR" "C:/opencv/build/x64/vc16/lib")
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(SudokuSolver Main.cpp)

target_link_libraries(SudokuSolver ${OpenCV_LIBS})