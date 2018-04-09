#include "RecordIOReader.hpp"
#include "TestRecordReader.hpp"
#include "common.hpp"
#include <stdexcept>
#include <exception>
#include "gmock/gmock.h"

using namespace sagemaker::tensorflow;

RecordReaderTest::RecordReaderTest() {}

RecordReaderTest::~RecordReaderTest() {}

void RecordReaderTest::SetUp() {}

void RecordReaderTest::TearDown() {}

class TestReader : RecordReader {
    public:
        TestReader(PipeReader& pipe_reader, std::size_t read_size, std::size_t buffer_capacity) 
            : RecordReader(pipe_reader, read_size, buffer_capacity) {}

        bool ReadRecord(std::string* storage) override {
            return false;
        }
        
        // make FillBuffer public for testing
        std::size_t WrapFillBuffer(std::size_t desired_size) {
            return FillBuffer(desired_size);
        }

        std::vector<char> GetBuffer() {
            return buffer_;
        }
};

TEST_F(RecordReaderTest, ValidConstructor) {
    std::string channel_dir = CreateTemporaryDirectory();
    CreateChannel(channel_dir, "elizabeth");
    PipeReader pipe_reader(channel_dir, "elizabeth");
    RecordIOReader reader(pipe_reader, 4, 32);
}

TEST_F(RecordReaderTest, InvalidConstructorSmallBuffer) {
    std::string channel_dir = CreateTemporaryDirectory();
    CreateChannel(channel_dir, "elizabeth");
    PipeReader pipe_reader(channel_dir, "elizabeth");
    
    EXPECT_THROW({
        RecordIOReader reader(pipe_reader, 4, 1);},
        std::invalid_argument);
}

TEST_F(RecordReaderTest, InvalidConstructorBufferLessThanReadSize) {
    std::string channel_dir = CreateTemporaryDirectory();
    CreateChannel(channel_dir, "elizabeth");
    PipeReader pipe_reader(channel_dir, "elizabeth");
    
    EXPECT_THROW({
        RecordIOReader reader(pipe_reader, 61, 60);},
        std::invalid_argument);
}

TEST_F(RecordReaderTest, InvalidConstructorReadSizeZero) {
    std::string channel_dir = CreateTemporaryDirectory();
    CreateChannel(channel_dir, "elizabeth");
    PipeReader pipe_reader(channel_dir, "elizabeth");
    
    EXPECT_THROW({
        RecordIOReader reader(pipe_reader, 1, 0);},
        std::invalid_argument);
}

#define VECTOR_EQ(expected, actual) { \
    ASSERT_EQ(expected.size(), reader.GetBuffer().size()) << "Vectors expected and target are of unequal length"; \
    for (int i = 0; i < expected.size(); ++i) { \
        EXPECT_EQ(expected[i], reader.GetBuffer()[i]) << "Vectors expected and target differ at index " << i; \
    } \
}

TEST_F(RecordReaderTest, CheckFillBuffer) {
    std::string channel_dir = CreateTemporaryDirectory();
    CreateChannel(channel_dir, "elizabeth", "0123456789ABCDEF", 0);
    PipeReader pipe_reader(channel_dir, "elizabeth");
    
    TestReader reader(pipe_reader, 2, 4);
    
    EXPECT_EQ(4, reader.WrapFillBuffer(4));
    std::vector<char> expected = {'0', '1', '2', '3'};
    VECTOR_EQ(expected, reader.GetBuffer());
    
    EXPECT_EQ(4, reader.WrapFillBuffer(4));
    expected = {'4', '5', '6', '7'};
    VECTOR_EQ(expected, reader.GetBuffer());

    EXPECT_EQ(4, reader.WrapFillBuffer(4));
    expected = {'8', '9', 'A', 'B'};
    VECTOR_EQ(expected, reader.GetBuffer());

    EXPECT_EQ(4, reader.WrapFillBuffer(4));
    expected = {'C', 'D', 'E', 'F'};
    VECTOR_EQ(expected, reader.GetBuffer());
}

TEST_F(RecordReaderTest, CheckFillUnevenReadSize) {
    std::string channel_dir = CreateTemporaryDirectory();
    CreateChannel(channel_dir, "elizabeth", "0123456789ABCDEF", 0);
    PipeReader pipe_reader(channel_dir, "elizabeth");
    
    TestReader reader(pipe_reader, 3, 4);
    
    EXPECT_EQ(4, reader.WrapFillBuffer(4));
    std::vector<char> expected = {'0', '1', '2', '3'};
    VECTOR_EQ(expected, reader.GetBuffer());
    
    EXPECT_EQ(4, reader.WrapFillBuffer(4));
    expected = {'4', '5', '6', '7'};
    VECTOR_EQ(expected, reader.GetBuffer());

    EXPECT_EQ(4, reader.WrapFillBuffer(4));
    expected = {'8', '9', 'A', 'B'};
    VECTOR_EQ(expected, reader.GetBuffer());

    EXPECT_EQ(4, reader.WrapFillBuffer(4));
    expected = {'C', 'D', 'E', 'F'};
    VECTOR_EQ(expected, reader.GetBuffer());
}

TEST_F(RecordReaderTest, DesiredSizeLessThanBufferSize) {
    std::string channel_dir = CreateTemporaryDirectory();
    CreateChannel(channel_dir, "elizabeth", "01234567", 0);
    PipeReader pipe_reader(channel_dir, "elizabeth");
    
    TestReader reader(pipe_reader, 3, 4);
    
    EXPECT_EQ(2, reader.WrapFillBuffer(2));
    std::vector<char> expected = {'0', '1'};
    VECTOR_EQ(expected, reader.GetBuffer());
    
    EXPECT_EQ(2, reader.WrapFillBuffer(2));
    expected = {'2', '3'};
    VECTOR_EQ(expected, reader.GetBuffer());

    EXPECT_EQ(2, reader.WrapFillBuffer(2));
    expected = {'4', '5'};
    VECTOR_EQ(expected, reader.GetBuffer());

    EXPECT_EQ(2, reader.WrapFillBuffer(2));
    expected = {'6', '7'};
    VECTOR_EQ(expected, reader.GetBuffer());
}

TEST_F(RecordReaderTest, DesiredSizeGreaterThanBufferSize) {
    std::string channel_dir = CreateTemporaryDirectory();
    CreateChannel(channel_dir, "elizabeth", "0123456789ABCDEF", 0);
    PipeReader pipe_reader(channel_dir, "elizabeth");
    
    TestReader reader(pipe_reader, 3, 4);
    
    EXPECT_EQ(4, reader.WrapFillBuffer(8));
    std::vector<char> expected = {'0', '1', '2', '3'};
    VECTOR_EQ(expected, reader.GetBuffer());
    
    EXPECT_EQ(4, reader.WrapFillBuffer(7));
    expected = {'4', '5', '6', '7'};
    VECTOR_EQ(expected, reader.GetBuffer());

    EXPECT_EQ(4, reader.WrapFillBuffer(6));
    expected = {'8', '9', 'A', 'B'};
    VECTOR_EQ(expected, reader.GetBuffer());

    EXPECT_EQ(4, reader.WrapFillBuffer(5));
    expected = {'C', 'D', 'E', 'F'};
    VECTOR_EQ(expected, reader.GetBuffer());
}

TEST_F(RecordReaderTest, CheckRepresentativeBufferSize) {
    std::string channel_dir = CreateTemporaryDirectory();

    std::size_t read_size = 64 * 1024;
    std::size_t desired_size = read_size * 2;
    std::size_t buffer_capacity = read_size * 128;
    std::size_t data_size = buffer_capacity * 2;
    
    char* data = new char[data_size];
    for (int i = 0; i < data_size; i++) {
        data [i] = "0123456789ABCDEF"[i % 16];
    }
    char* output = new char[desired_size];

    CreateChannel(channel_dir, "elizabeth", data, 0);

    PipeReader pipe_reader(channel_dir, "elizabeth");
    TestReader reader(pipe_reader, read_size, buffer_capacity);
    
    // Should be able to do 64 reads
    for (int i = 0; i < buffer_capacity / desired_size; i += desired_size) {
        EXPECT_EQ(desired_size, reader.WrapFillBuffer(desired_size));
    }
    delete [] data;
    delete [] output;
}
