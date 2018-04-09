#ifndef RECORD_IO_READER_H
#define RECORD_IO_READER_H

#include <RecordReader.hpp>

namespace sagemaker {
namespace tensorflow {
    
    class RecordIOReader final : RecordReader {

    public:
        explicit RecordIOReader(PipeReader& pipe_reader, std::size_t read_size, std::size_t buffer_capacity);
        bool ReadRecord(std::string* storage) override;
    private:
        std::size_t read_size_;
    }; 
} // sagemaker
} // tensorflow
#endif
