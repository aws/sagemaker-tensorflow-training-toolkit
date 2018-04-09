file(GLOB_RECURSE sources ./*.cpp ./*.hpp)

add_executable(testRecordIOReader ${sources})
target_include_directories(testRecordIOReader PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(testRecordIOReader
    libPipeReader
    libRecordIOReader
    libgtest
    libgmock
)

add_test(NAME testRecordIOReader COMMAND testRecordIOReader)