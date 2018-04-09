cmake_minimum_required(VERSION 3.10)
set (CMAKE_CXX_STANDARD 11)

file(GLOB_RECURSE sources ./*.cpp ./*.hpp)
add_library(libPipeReader STATIC ${sources})

target_include_directories(libPipeReader PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)