#ifndef SAGEMAKER_TENSORFLOW_COMMON_H
#define SAGEMAKER_TENSORFLOW_COMMON_H
#include <string>

std::string CreateTemporaryDirectory(); 
void CreateChannel(const std::string& channel_directory, const std::string& channel_name, const char* data, unsigned int index);
void CreateChannel(const std::string& channel_directory, const std::string& channel_name);

#endif