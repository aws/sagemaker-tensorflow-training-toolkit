#ifndef SAGEMAKER_PIPE_READER_H
#define SAGEMAKER_PIPE_READER_H

#include <cstdio>
#include <string>
#include <unistd.h>

namespace sagemaker { 
namespace tensorflow {

    class PipeReader {
    public:
        /**
           Constructs a new PipeReader. 

           The PipeReader reads data from the SageMaker Channel channelName, as a named pipe in
           channelDirectory. This PipeReader may be immediately Read after construction.

           Objects of this class implement the SageMaker PipeMode channel protocol. The entire channel data
           may be read sequentially by invoking Read. The reader can be reset to read from the beginning of the
           channel by invoking Reset. This can occur when the Reader has reached the end of the channel data or before. 
           At anytime Close may be called to release the underlying filesystem resources. Reset can be invoked after Close,
           to begin reginning from the channel again.

           @param[in] channelDirectory The Posix filesystem directory where SageMaker channel named pipes 
                                       are created.
           @param[in] channelName The name of the channel to read.
        */
        PipeReader(const std::string & channel_directory, const std::string & channel_name); 
        
        PipeReader(const PipeReader& other) = delete;
        PipeReader& operator=(const PipeReader&) = delete;
        PipeReader(PipeReader&& other);
        PipeReader& operator=(PipeReader&& other);

        /**
           Destructs a PipeReader. Invokes Close.
        */
        ~PipeReader();

        /**
           Reads up-to size bytes from the current pipe into buffer. Returns the number of bytes read. Will 
           return 0 if there is no data left to read or size was 0.

           Raises system_error with errno set on read error.

           @param[out] buffer The buffer to read data into.
           @param[in]  size The desired number of bytes to be read.
           @return The number of bytes read.
           
        */
        std::size_t Read(void* buffer, std::size_t size);

        /**
           Advances the reader to the next channel pipe.
        */
        void Reset();
        
        /**
           Closes the current pipe. Subsequent calls to Read will raise TODO:Exception, unless a call
           to Reset is first made.
        */
        void Close();
        
        /**
           Returns the directory where channel pipes are read.
        */
        std::string GetChannelDirectory() const {
            return channel_directory_;
        }

        /**
           Returns the SageMaker Pipe Mode channel name that is being read by this PipeReader.
          */
        std::string GetChannelName() const {
            return channel_name_;
        }

    private:
        std::string BuildCurrentPipeName() const;
        void Open();
        std::uint32_t current_pipe_index_;
        int current_pipe_;
        std::string channel_directory_;
        std::string channel_name_;
    };
} // tensorflow
} // sagemaker

#endif