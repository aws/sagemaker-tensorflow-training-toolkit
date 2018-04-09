#include "common.hpp"
#include <cstdlib>
#include <fstream>

std::string CreateTemporaryDirectory() {
    char mkdTemplate[] = "/tmp/tmpdir.XXXXXX";
    return std::string(mkdtemp(mkdTemplate));
}

void CreateChannel(const std::string& channel_directory, const std::string& channel_name, 
    const char* data, unsigned int index) {
    
    std::string pipe_name = channel_name + "_" + std::to_string(index);
    std::string channel_path = channel_directory;
    if (channel_path[channel_path.length() - 1] != '/') {
        channel_path += '/';
    }
    channel_path += pipe_name;

    std::ofstream file(channel_path); 
    file << data << '\0';
}

void CreateChannel(const std::string& channel_directory, const std::string& channel_name) {  
    CreateChannel(channel_directory, channel_name, "blah", 0);
}
