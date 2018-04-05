#include "PipeReader.hpp"
#include "TestPipeReader.hpp"
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>

std::string createTemporaryDirectory() {
    char mkdTemplate[] = "/tmp/tmpdir.XXXXXX";
    return std::string(mkdtemp(mkdTemplate));
}

void createChannel(const std::string& channelDirectory, const std::string& channelName, const char* data, unsigned int index) {
    std::string pipeName = channelName + "_" + std::to_string(index);
    std::string channelPath = channelDirectory;
    if (channelPath[channelPath.length() - 1] != '/') {
        channelPath += '/';
    }
    channelPath += pipeName;

    std::ofstream file(channelPath); 
    file << data << '\0';
}

void createChannel(const std::string& channelDirectory, const std::string& channelName) {
    createChannel(channelDirectory, channelName, "blah", 0);
}


PipeReaderTest::PipeReaderTest() {}

PipeReaderTest::~PipeReaderTest() {};

void PipeReaderTest::SetUp() {};

void PipeReaderTest::TearDown() {};

using namespace sagemaker;

TEST_F(PipeReaderTest, CanOpenFile) {
    std::string channelDirectory = createTemporaryDirectory();
    createChannel(channelDirectory, "elizabeth");
    PipeReader reader(channelDirectory, "elizabeth");
}

TEST_F(PipeReaderTest, GetDirectory) {
    std::string channelDirectory = createTemporaryDirectory();
    createChannel(channelDirectory, "elizabeth");
    PipeReader reader(channelDirectory, "elizabeth");
    EXPECT_EQ(reader.GetChannelDirectory(), channelDirectory);
}

TEST_F(PipeReaderTest, GetChannel) {
    std::string channelDirectory = createTemporaryDirectory();
    createChannel(channelDirectory, "elizabeth");
    PipeReader reader(channelDirectory, "elizabeth");
    EXPECT_EQ(reader.GetChannelName(), "elizabeth");
}

TEST_F(PipeReaderTest, ReadPipe) {
    std::string channelDirectory = createTemporaryDirectory();
    createChannel(channelDirectory, "elizabeth", "abc", 0);
    PipeReader reader(channelDirectory, "elizabeth");
    char buffer [4]; // 3 + 1 for the \0
    size_t readNum = reader.Read(static_cast<void*>(buffer), 4);
    EXPECT_STREQ("abc", buffer);
    EXPECT_EQ(4, readNum);
}

TEST_F(PipeReaderTest, ReadPipeAfterEOF) {
    std::string channelDirectory = createTemporaryDirectory();
    createChannel(channelDirectory, "elizabeth", "abc", 0);
    PipeReader reader(channelDirectory, "elizabeth");
    char buffer [4];
    size_t readNum = reader.Read(static_cast<void*>(buffer), 4);
    readNum = reader.Read(static_cast<void*>(buffer), 1);
    EXPECT_EQ(0, readNum);
}

TEST_F(PipeReaderTest, Reset) {
    std::string channelDirectory = createTemporaryDirectory();
    createChannel(channelDirectory, "elizabeth", "abc", 0);
    PipeReader reader(channelDirectory, "elizabeth");
    char buffer [4]; 
    size_t readNum = reader.Read(static_cast<void*>(buffer), 4);
    
    // Create second iteration of elizabeth
    createChannel(channelDirectory, "elizabeth", "def", 1);
    reader.Reset();
    readNum = reader.Read(static_cast<void*>(buffer), 4);
    EXPECT_STREQ("def", buffer);
    EXPECT_EQ(4, readNum);
}

TEST_F(PipeReaderTest, CloseReset) {
    std::string channelDirectory = createTemporaryDirectory();
    createChannel(channelDirectory, "elizabeth", "abc", 0);
    PipeReader reader(channelDirectory, "elizabeth");
    char buffer [4]; 
    
    // Read partially through the channel's current data
    size_t readNum = reader.Read(static_cast<void*>(buffer), 2);
    
    // Close the current pipe
    reader.Close();

    // Create second iteration of elizabeth
    createChannel(channelDirectory, "elizabeth", "def", 1);
    reader.Reset();
    readNum = reader.Read(static_cast<void*>(buffer), 4);
    EXPECT_STREQ("def", buffer);
    EXPECT_EQ(4, readNum);
}

TEST_F(PipeReaderTest, CloseImmediatelyReset) {
    std::string channelDirectory = createTemporaryDirectory();
    createChannel(channelDirectory, "elizabeth", "abc", 0);
    PipeReader reader(channelDirectory, "elizabeth");
    char buffer [4]; 
    
    // Close the current pipe, without reading anything from it
    reader.Close();

    // Create second iteration of elizabeth
    createChannel(channelDirectory, "elizabeth", "def", 1);
    reader.Reset();
    size_t readNum = reader.Read(static_cast<void*>(buffer), 4);
    EXPECT_STREQ("def", buffer);
    EXPECT_EQ(4, readNum);
}
