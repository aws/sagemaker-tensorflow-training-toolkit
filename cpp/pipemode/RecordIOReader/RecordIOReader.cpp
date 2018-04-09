#include <exception>
#include <stdexcept>
#include <cstring>
#include <RecordIOReader.hpp>

using namespace sagemaker::tensorflow;

std::uint32_t RECORD_IO_MAGIC = 0xced7230a;

struct RecordIOHeader {
    std::uint32_t magic_number;
    std::uint32_t size_and_flag;
};

inline void ValidateMagicNumber(RecordIOHeader& header) {
    if(header.magic_number != RECORD_IO_MAGIC) {
        throw std::runtime_error("Invalid magic number");
    }
}

inline std::uint32_t GetRecordSize(RecordIOHeader& header) {
    return header.size_and_flag & ((1u << 29u) - 1u);
}

inline std::uint32_t GetRecordFlag(RecordIOHeader& header) {
    return (header.size_and_flag >> 29u) & 7u;
}

inline std::uint32_t GetPaddedSize(std::uint32_t size) {
    return size + (4 - size % 4) % 4;
}

RecordIOReader::RecordIOReader(PipeReader& pipe_reader, std::size_t read_size, std::size_t buffer_capacity) 
    : RecordReader(pipe_reader, read_size, buffer_capacity) {}

bool RecordIOReader::ReadRecord(std::string* storage) {
    RecordIOHeader header;
    std::size_t bytes_read = pipe_reader_.Read(&header, sizeof(header));
    if (! bytes_read) {
        return false;
    }
    ValidateMagicNumber(header);
    if(0 != GetRecordFlag(header)) { // RecordIO multipart records are not yet supported.
        throw std::runtime_error("Multipart records are not supported");
    }

    std::size_t expected_size = GetRecordSize(header);
    std::size_t padded_expected_size = GetPaddedSize(expected_size);
    
    storage->resize(expected_size);
    for (std::size_t i = 0; i < padded_expected_size; i += buffer_.capacity()) {
        std::size_t bytes_read = FillBuffer(padded_expected_size - i);
        std::size_t write_bytes = std::min(bytes_read, expected_size - i);
        std::memcpy(&(storage->at(0)) + i, buffer_.data(), write_bytes);
    }
    return true;
}
