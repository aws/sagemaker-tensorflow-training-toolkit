file(GLOB SRCS *.cpp)

add_executable(testPipeReader ${SRCS})

target_link_libraries(testPipeReader
    libPipeReader
    libgtest
    libgmock
)

add_test(NAME testPipeReader COMMAND testPipeReader)