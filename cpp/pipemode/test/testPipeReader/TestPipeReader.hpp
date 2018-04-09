#ifndef SAGEMAKER_TENSORFLOW_PIPE_READER_TEST_H
#define SAGEMAKER_TENSORFLOW_PIPE_READER_TEST_H
#include "gtest/gtest.h"

namespace sagemaker {
namespace tensorflow {
class PipeReaderTest : public ::testing::Test {

protected:

    // You can do set-up work for each test here.
    PipeReaderTest();

    // You can do clean-up work that doesn't throw exceptions here.
    virtual ~PipeReaderTest();

    // If the constructor and destructor are not enough for setting up
    // and cleaning up each test, you can define the following methods:

    // Code here will be called immediately after the constructor (right
    // before each test).
    virtual void SetUp();

    // Code here will be called immediately after each test (right
    // before the destructor).
    virtual void TearDown();

};
} // tensorflow
} // sagemaker
#endif