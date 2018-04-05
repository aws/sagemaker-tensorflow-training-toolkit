#include "PipeReader.hpp"
#include <system_error>
#include <fcntl.h>

using namespace sagemaker;

PipeReader::PipeReader(const std::string & channelDirectory, const std::string & channelName) : 
            channelDirectory(channelDirectory), 
            channelName(channelName), 
            currentPipeIndex(0),
            currentPipe(-1) {
    Open();
}

PipeReader::~PipeReader() {
    Close();
}

void PipeReader::Open() {
    currentPipe = open(BuildCurrentPipeName().c_str(), O_RDONLY);
    if(-1 == currentPipe) {
        throw std::system_error(errno, std::system_category());
    }
}

void PipeReader::Close() {
    if(-1 != currentPipe) {
        close(currentPipe);
        currentPipe = -1;
    }
}

void PipeReader::Reset() {
    Close();
    currentPipeIndex ++;
    Open();
}

std::size_t PipeReader::Read(void* buffer, std::size_t size) {
    ssize_t nbytes = read(currentPipe, buffer, size);
    if(nbytes < 0) {
        throw std::system_error(errno, std::system_category());
    }
    return static_cast<size_t>(nbytes);
}

std::string PipeReader::BuildCurrentPipeName() const {
    std::string pipeName = channelName + "_" + std::to_string(currentPipeIndex);
    std::string channelPath = channelDirectory;
    if (channelPath[channelPath.length() - 1] != '/') {
        channelPath += '/';
    }
    channelPath += pipeName;
    return channelPath;
}
