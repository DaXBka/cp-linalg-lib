cmake_minimum_required(VERSION 3.26)
project(cp_linalg_lib)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_CLANG_TIDY "clang-tidy;-p=${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(main main.cpp)

include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
enable_testing()

add_executable(test_matrix tests/test_matrix.cpp)
target_link_libraries(test_matrix PUBLIC gtest_main)

include(GoogleTest)
gtest_discover_tests(test_matrix)
