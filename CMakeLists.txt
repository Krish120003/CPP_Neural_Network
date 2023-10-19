cmake_minimum_required(VERSION 3.0.0)
project(datasci_project VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
include_directories("./include/")

file(GLOB TARGET_SRC "./src/*.cpp")

add_executable(main ${TARGET_SRC})
