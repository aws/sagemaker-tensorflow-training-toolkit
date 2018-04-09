cmake_minimum_required(VERSION 3.10)
set (CMAKE_CXX_STANDARD 17)

file(GLOB_RECURSE sources ./*.cpp ./*.hpp)
add_library(libRecordIOReader STATIC ${sources})

target_link_libraries(libRecordIOReader libPipeReader)

target_include_directories(libRecordIOReader PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)