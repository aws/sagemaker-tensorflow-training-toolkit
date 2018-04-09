#ifndef RECORD_READER_H
#define RECORD_READER_H

#include <vector>
#include <string>
#include <PipeReader.hpp>

namespace sagemaker {
namespace tensorflow {

/**
   An abstract record reader that reads records from a PipeReader.
  */
class RecordReader {

public:

	/**
	   Constructs a new RecordReader.

	   Records are read from the specified PipeReader reader. Each call to read on reader reads
	   read_size bytes. The RecordReader buffers up to buffer_capacity bytes from the underlying 
	   PipeReader
	*/
    RecordReader(PipeReader& reader, const std::size_t read_size, const std::size_t buffer_capacity);
    
    /**
       Reads a record from the underlying PipeReader and stores the record data in the 
       specified string pointer. The specified string is resized to accomodate the record.

       param [out] storage The string where the record is written to.
       return true if a record could be read, false otherwise.
    */
    virtual bool ReadRecord(std::string* storage) = 0;

protected:

	/**
	   Reads up-to desired size bytes from this RecordReader's PipeReader. Returns the
	   number of bytes read.

       param[in] desired_size The number of bytes to read.
       return The number of bytes actually read.
	*/
    std::size_t FillBuffer(std::size_t desired_size);

    /**
       The current pipe reader of this RecordReader
      */
    PipeReader& pipe_reader_;
    
    /**
       The buffer of characters read from pipe_reader_
      */
    std::vector<char> buffer_;

private:
    std::size_t read_size_;
    std::size_t buffer_capacity_;
};
} // tensorflow
} // sagemaker

#endif