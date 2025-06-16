//
// Created by asus on 25-6-4.
//
#include <gtest/gtest.h>
#include <glog/logging.h>

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    google::InitGoogleLogging("KXInfer");
    FLAGS_log_dir = "/home/asus/CLionProjects/KXInfer/log/";
    FLAGS_alsologtostderr = true;

    LOG(INFO) << "Start Test...\n";
    return RUN_ALL_TESTS();
}