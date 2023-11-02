cmake_minimum_required(VERSION 3.0.0)
project(datasci_project VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

include_directories("./include/")

file(GLOB TARGET_SRC "./src/main.cpp")
file(GLOB TEST_TARGET_SRC "./src/test.cpp")
file(GLOB XOR_NET "./src/learn_xor.cpp")

add_executable(main ${TARGET_SRC})
add_executable(test ${TEST_TARGET_SRC})
add_executable(xor ${XOR_NET})
