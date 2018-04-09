#include "TestRecordIOReader.hpp"
#include "PipeReader.hpp"
#include "RecordIOReader.hpp"
#include <sstream>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <stdexcept>

using namespace sagemaker::tensorflow;

RecordIOReaderTest::RecordIOReaderTest() {}

RecordIOReaderTest::~RecordIOReaderTest() {};

void RecordIOReaderTest::SetUp() {};

void RecordIOReaderTest::TearDown() {};

std::string CreateTemporaryDirectory() {
    char mkdTemplate[] = "/tmp/tmpdir.XXXXXX";
    return std::string(mkdtemp(mkdTemplate));
}

void CreateChannel(const std::string& channel_directory, const std::string& channel_name, const std::string data, unsigned int index) {
    std::string pipe_name = channel_name + "_" + std::to_string(index);
    std::string channel_path = channel_directory;
    if (channel_path[channel_path.length() - 1] != '/') {
        channel_path += '/';
    }
    channel_path += pipe_name;
    std::ofstream file(channel_path); 
    file.write(data.data(), data.size());
}

void CreateChannel(const std::string& channel_directory, const std::string& channel_name) {  
    CreateChannel(channel_directory, channel_name, "blah", 0);
}

std::string ToRecordIO(const std::string& data) {
    std::vector<char> vec(8);;
    
    vec[0] = 0xa;
    vec[1] = 0x23;
    vec[2] = 0xd7;
    vec[3] = 0xce;

    std::uint32_t length = data.size();
    char* plength = (char*)&length;

    vec[4] = *(plength + 0);
    vec[5] = *(plength + 1);
    vec[6] = *(plength + 2);
    vec[7] = *(plength + 3);

    vec.insert(vec.end(), data.begin(), data.end());

    std::uint32_t padding = (4 - length % 4) % 4;
    for (int i = 0; i < padding; i++) {
        vec.push_back(' ');
    }
    std::string encoding;
    encoding.insert(encoding.begin(), vec.begin(), vec.end());
    return encoding;
}

TEST_F(RecordIOReaderTest, InvalidMagicNumber) {
    std::string channel_dir = CreateTemporaryDirectory();
    CreateChannel(channel_dir, "elizabeth", "not a magic number", 0);
    PipeReader pipe_reader(channel_dir, "elizabeth");
    RecordIOReader reader(pipe_reader, 4, 4);
    std::string storage;
    
    EXPECT_THROW({
        reader.ReadRecord(&storage);},
        std::runtime_error);
}


TEST_F(RecordIOReaderTest, InvalidHeader) {
    std::string channel_dir = CreateTemporaryDirectory();

    std::vector<char> vec(8);;
    
    vec[0] = 0xa;
    vec[1] = 0x23;
    vec[2] = 0xd7;
    vec[3] = 0xce;

    std::string data = "abcd";
    std::uint32_t length = data.size();
    length |= (1u << 29u);
    char* plength = (char*)&length;

    vec[4] = *(plength + 0);
    vec[5] = *(plength + 1);
    vec[6] = *(plength + 2);
    vec[7] = *(plength + 3);
    
    vec.insert(vec.end(), data.begin(), data.end());
    std::string encoding;
    encoding.insert(encoding.begin(), vec.begin(), vec.end());

    CreateChannel(channel_dir, "elizabeth", encoding, 0);
    PipeReader pipe_reader(channel_dir, "elizabeth");
    RecordIOReader reader(pipe_reader, 4, 4);
    std::string storage;

    EXPECT_THROW({
        reader.ReadRecord(&storage);},
        std::runtime_error);
}



TEST_F(RecordIOReaderTest, TestReadSingleRecord) {
    std::string channel_dir = CreateTemporaryDirectory();
    std::string input = "Elizabeth Is 10 months Old";
    std::string encoded = ToRecordIO(input);
    CreateChannel(channel_dir, "elizabeth", encoded, 0);
    PipeReader pipe_reader(channel_dir, "elizabeth");
    RecordIOReader reader(pipe_reader, 4, 4);
    std::string result;
    reader.ReadRecord(&result);
    EXPECT_EQ(input, result);
}

TEST_F(RecordIOReaderTest, TestReadMultipleRecords) {
    std::string channel_dir = CreateTemporaryDirectory();
    std::string input = "Elizabeth";
    std::string multi_record;
    for (int i = 0; i < 10; i++) {
        multi_record += ToRecordIO(input + std::to_string(i));
    }
    CreateChannel(channel_dir, "elizabeth", multi_record, 0);
    PipeReader pipe_reader(channel_dir, "elizabeth");
    RecordIOReader reader(pipe_reader, 4, 4);
    for(int i = 0; i < 10; i++) {
        std::string result;
        reader.ReadRecord(&result);
        EXPECT_EQ(input + std::to_string(i), result);
    }
}

