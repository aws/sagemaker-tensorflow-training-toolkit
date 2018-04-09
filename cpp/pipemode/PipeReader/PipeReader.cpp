#include "PipeReader.hpp"
#include <system_error>
#include <fcntl.h>
#include <iostream>

using namespace sagemaker::tensorflow;

PipeReader::PipeReader(const std::string & channel_directory, const std::string & channel_name) : 
            channel_directory_(channel_directory), 
            channel_name_(channel_name), 
            current_pipe_index_(0),
            current_pipe_(-1) {
    Open();
}

PipeReader::PipeReader(PipeReader&& other) :
            channel_directory_(other.channel_directory_),
            channel_name_(other.channel_name_),
            current_pipe_index_(other.current_pipe_index_) {
    Open();
    close(other.current_pipe_);
    other.current_pipe_ = -1;
}

PipeReader& PipeReader::operator=(PipeReader&& other) {
    Close();
    channel_directory_ = std::move(other.channel_directory_);
    channel_name_ = std::move(other.channel_name_);
    current_pipe_index_ = std::move(other.current_pipe_index_);
    Open();
    close(other.current_pipe_);
    other.current_pipe_ = -1;
    return *this;
}

PipeReader::~PipeReader() {
    Close();
}

void PipeReader::Open() {
    current_pipe_ = open(BuildCurrentPipeName().c_str(), O_RDONLY);
    if(-1 == current_pipe_) {
        throw std::system_error(errno, std::system_category());
    }
}

void PipeReader::Close() {
    if(-1 != current_pipe_) {
        close(current_pipe_);
        current_pipe_ = -1;
    }
}

void PipeReader::Reset() {
    Close();
    current_pipe_index_ ++;
    Open();
}

std::size_t PipeReader::Read(void* buffer, std::size_t size) {
    ssize_t nbytes = read(current_pipe_, buffer, size);
    if(nbytes < 0) {
        throw std::system_error(errno, std::system_category());
    }
    return static_cast<size_t>(nbytes);
}

std::string PipeReader::BuildCurrentPipeName() const {
    std::string pipe_name = channel_name_ + "_" + std::to_string(current_pipe_index_);
    std::string channel_path = channel_directory_;
    if (channel_path[channel_path.length() - 1] != '/') {
        channel_path += '/';
    }
    channel_path += pipe_name;
    return channel_path;
}
