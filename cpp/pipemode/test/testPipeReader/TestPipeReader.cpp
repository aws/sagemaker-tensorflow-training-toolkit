#include "PipeReader.hpp"
#include "TestPipeReader.hpp"
#include "common.hpp"
#include <stdio.h>
using namespace sagemaker::tensorflow;

PipeReaderTest::PipeReaderTest() {}

PipeReaderTest::~PipeReaderTest() {};

void PipeReaderTest::SetUp() {};

void PipeReaderTest::TearDown() {};

PipeReader MakeReader(std::string channelDirectory) {
    CreateChannel(channelDirectory, "elizabeth", "abc", 0);
    PipeReader reader(channelDirectory, "elizabeth");
    return reader;
}

TEST_F(PipeReaderTest, TestMoveConstructor) {
    std::string dir = CreateTemporaryDirectory();
    PipeReader reader(MakeReader(dir));
    EXPECT_EQ("elizabeth", reader.GetChannelName());
    EXPECT_EQ(dir, reader.GetChannelDirectory());
    char buffer [4]; // 3 + 1 for the \0
    size_t readNum = reader.Read(static_cast<void*>(buffer), 4);
    EXPECT_STREQ("abc", buffer);
    EXPECT_EQ(4, readNum);
}


TEST_F(PipeReaderTest, TestAssignment) {
    std::string channelDirectory = CreateTemporaryDirectory();
    CreateChannel(channelDirectory, "elizabeth", "abc", 0);
    CreateChannel(channelDirectory, "not_elizabeth", "def", 0);
    PipeReader reader(channelDirectory, "not_elizabeth");
    std::string dir = CreateTemporaryDirectory();
    reader = MakeReader(dir);
    EXPECT_EQ("elizabeth", reader.GetChannelName());
    EXPECT_EQ(dir, reader.GetChannelDirectory());
    char buffer [4]; // 3 + 1 for the \0
    size_t readNum = reader.Read(static_cast<void*>(buffer), 4);
    EXPECT_STREQ("abc", buffer);
    EXPECT_EQ(4, readNum);
}


TEST_F(PipeReaderTest, CanOpenFile) {
    std::string channelDirectory = CreateTemporaryDirectory();
    CreateChannel(channelDirectory, "elizabeth");
    PipeReader reader(channelDirectory, "elizabeth");
}

TEST_F(PipeReaderTest, GetDirectory) {
    std::string channelDirectory = CreateTemporaryDirectory();
    CreateChannel(channelDirectory, "elizabeth");
    PipeReader reader(channelDirectory, "elizabeth");
    EXPECT_EQ(reader.GetChannelDirectory(), channelDirectory);
}

TEST_F(PipeReaderTest, GetChannel) {
    std::string channelDirectory = CreateTemporaryDirectory();
    CreateChannel(channelDirectory, "elizabeth");
    PipeReader reader(channelDirectory, "elizabeth");
    EXPECT_EQ(reader.GetChannelName(), "elizabeth");
}

TEST_F(PipeReaderTest, ReadPipe) {
    std::string channelDirectory = CreateTemporaryDirectory();
    CreateChannel(channelDirectory, "elizabeth", "abc", 0);
    PipeReader reader(channelDirectory, "elizabeth");
    char buffer [4]; // 3 + 1 for the \0
    size_t readNum = reader.Read(static_cast<void*>(buffer), 4);
    EXPECT_STREQ("abc", buffer);
    EXPECT_EQ(4, readNum);
}

TEST_F(PipeReaderTest, ReadPipeAfterEOF) {
    std::string channelDirectory = CreateTemporaryDirectory();
    CreateChannel(channelDirectory, "elizabeth", "abc", 0);
    PipeReader reader(channelDirectory, "elizabeth");
    char buffer [4];
    size_t readNum = reader.Read(static_cast<void*>(buffer), 4);
    readNum = reader.Read(static_cast<void*>(buffer), 1);
    EXPECT_EQ(0, readNum);
}

TEST_F(PipeReaderTest, Reset) {
    std::string channelDirectory = CreateTemporaryDirectory();
    CreateChannel(channelDirectory, "elizabeth", "abc", 0);
    PipeReader reader(channelDirectory, "elizabeth");
    char buffer [4]; 
    size_t readNum = reader.Read(static_cast<void*>(buffer), 4);
    
    // Create second iteration of elizabeth
    CreateChannel(channelDirectory, "elizabeth", "def", 1);
    reader.Reset();
    readNum = reader.Read(static_cast<void*>(buffer), 4);
    EXPECT_STREQ("def", buffer);
    EXPECT_EQ(4, readNum);
}

TEST_F(PipeReaderTest, CloseReset) {
    std::string channelDirectory = CreateTemporaryDirectory();
    CreateChannel(channelDirectory, "elizabeth", "abc", 0);
    PipeReader reader(channelDirectory, "elizabeth");
    char buffer [4]; 
    
    // Read partially through the channel's current data
    size_t readNum = reader.Read(static_cast<void*>(buffer), 2);
    
    // Close the current pipe
    reader.Close();

    // Create second iteration of elizabeth
    CreateChannel(channelDirectory, "elizabeth", "def", 1);
    reader.Reset();
    readNum = reader.Read(static_cast<void*>(buffer), 4);
    EXPECT_STREQ("def", buffer);
    EXPECT_EQ(4, readNum);
}

TEST_F(PipeReaderTest, CloseImmediatelyReset) {
    std::string channelDirectory = CreateTemporaryDirectory();
    CreateChannel(channelDirectory, "elizabeth", "abc", 0);
    PipeReader reader(channelDirectory, "elizabeth");
    char buffer [4]; 
    
    // Close the current pipe, without reading anything from it
    reader.Close();

    // Create second iteration of elizabeth
    CreateChannel(channelDirectory, "elizabeth", "def", 1);
    reader.Reset();
    size_t readNum = reader.Read(static_cast<void*>(buffer), 4);
    EXPECT_STREQ("def", buffer);
    EXPECT_EQ(4, readNum);
}

TEST_F(PipeReaderTest, OpenFails) {
    EXPECT_THROW({
        PipeReader reader("I am not a directory #$%^?!", "blah");},
        std::system_error);
}
