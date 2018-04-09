cmake_minimum_required(VERSION 3.10)
enable_language(CXX)
set(CMAKE_CXX_STANDARD 11)
set(CURL_LIBRARY "-lcurl") 
find_package(CURL REQUIRED) 
project(TFPipeModeDataset)

enable_testing()

add_subdirectory(PipeReader)
add_subdirectory(RecordIOReader)
add_subdirectory(test)
add_executable(go go.cpp)




add_executable(in in.cpp)
target_include_directories(in PUBLIC
    PipeReader
    RecordIOReader
)
target_link_libraries(in
    libPipeReader
    libRecordIOReader)