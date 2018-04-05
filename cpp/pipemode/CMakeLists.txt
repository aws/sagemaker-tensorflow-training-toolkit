cmake_minimum_required(VERSION 3.0)
enable_language(CXX)
set(CMAKE_CXX_STANDARD 11)
set(CURL_LIBRARY "-lcurl") 
find_package(CURL REQUIRED) 
project(TFPipeModeDataset)

enable_testing()

add_subdirectory(PipeReader)
add_subdirectory(test)
