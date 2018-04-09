file(GLOB_RECURSE sources ./*.cpp ./*.hpp)

add_executable(testPipeReader ${sources})
target_include_directories(testPipeReader PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(testPipeReader
    libPipeReader
    libRecordIOReader
    libgtest
    libgmock
)

add_test(NAME testPipeReader COMMAND testPipeReader)