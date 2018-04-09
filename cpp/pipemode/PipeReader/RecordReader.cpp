#include <RecordReader.hpp>
#include <stdexcept>

using namespace sagemaker::tensorflow;

RecordReader::RecordReader(PipeReader& reader, const std::size_t read_size, 
    const std::size_t buffer_capacity) : read_size_(read_size), pipe_reader_(reader), 
    buffer_(buffer_capacity), buffer_capacity_(buffer_capacity) {
    if (buffer_capacity < 4) {
        throw std::invalid_argument("buffer_capacity must be at least 4");
    }
    if (buffer_capacity < read_size) {
        throw std::invalid_argument("buffer_capacity must be larger than read_size");
    }
    if (0 == read_size) {
        throw std::invalid_argument("read_size must be greater-than zero");
    }
}

std::size_t RecordReader::FillBuffer(std::size_t desired_size) {
    buffer_.resize(buffer_capacity_);
    std::size_t fill_read_size = std::min(buffer_capacity_, desired_size);
    std::size_t bytes_read = 0;
    while(bytes_read < fill_read_size) {
        
        std::size_t next_read_size = pipe_reader_.Read(buffer_.data() + bytes_read, 
            std::min(fill_read_size, desired_size - bytes_read));
        
        if(! next_read_size) {
            break;
        }
        bytes_read += next_read_size;
    }
    buffer_.resize(bytes_read);
    return bytes_read;
}
