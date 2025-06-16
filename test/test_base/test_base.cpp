//
// Created by asus on 25-6-4.
//
#include <iostream>
#include <sstream>
#include "base/base.h"
#include <gtest/gtest.h>
#include<glog/logging.h>

// 可选：定义一个初始化日志的宏，控制是否将日志输出到 stderr

TEST(StatusTest, SuccessCheck) {
    LOG(INFO) << "Testing base::error::Success";

    base::Status s = base::error::Success("操作成功");
    LOG(INFO) << "Status created: " << s;

    EXPECT_TRUE(s);
    EXPECT_EQ(s.get_err_code(), base::kSuccess);
    EXPECT_EQ(s.get_err_msg(), "操作成功");
    EXPECT_EQ(s, base::kSuccess);
    EXPECT_NE(s, base::kInvalidArgument);

    // 测试 operator<< 输出
    std::stringstream ss;
    ss << s;
    EXPECT_EQ(ss.str(), "操作成功");

    LOG(INFO) << "SuccessCheck test passed.";
}

TEST(StatusTest, InvalidArgumentError) {
    LOG(INFO) << "Testing base::error::InvalidArgument";

    base::Status s = base::error::InvalidArgument("参数错误");
    LOG(INFO) << "Status created: " << s;

    EXPECT_FALSE(s);
    EXPECT_EQ(s.get_err_code(), base::kInvalidArgument);
    EXPECT_EQ(s.get_err_msg(), "参数错误");
    EXPECT_NE(s, base::kSuccess);
    EXPECT_EQ(s, base::kInvalidArgument);

    LOG(INFO) << "InvalidArgumentError test passed.";
}

TEST(StatusTest, InternalError) {
    LOG(INFO) << "Testing base::error::InternalError";

    base::Status s = base::error::InternalError("内部错误");
    LOG(INFO) << "Status created: " << s;

    EXPECT_FALSE(s);
    EXPECT_EQ(s.get_err_code(), base::kInternalError);
    EXPECT_EQ(s.get_err_msg(), "内部错误");

    LOG(INFO) << "InternalError test passed.";
}

TEST(StatusTest, PathNotValid) {
    LOG(INFO) << "Testing base::error::PathNotValid";

    base::Status s = base::error::PathNotValid("路径无效");
    LOG(INFO) << "Status created: " << s;

    EXPECT_FALSE(s);
    EXPECT_EQ(s.get_err_code(), base::kPathNotValid);
    EXPECT_EQ(s.get_err_msg(), "路径无效");

    LOG(INFO) << "PathNotValid test passed.";
}

TEST(StatusTest, ModelParseError) {
    LOG(INFO) << "Testing base::error::ModelParseError";

    base::Status s = base::error::ModelParseError("模型解析失败");
    LOG(INFO) << "Status created: " << s;

    EXPECT_FALSE(s);
    EXPECT_EQ(s.get_err_code(), base::kModelParseError);
    EXPECT_EQ(s.get_err_msg(), "模型解析失败");

    LOG(INFO) << "ModelParseError test passed.";
}

TEST(StatusTest, FunctionNotImplement) {
    LOG(INFO) << "Testing base::error::FunctionNotImplement";

    base::Status s = base::error::FunctionNotImplement("功能未实现");
    LOG(INFO) << "Status created: " << s;

    EXPECT_FALSE(s);
    EXPECT_EQ(s.get_err_code(), base::kFunctionUnImplement);
    EXPECT_EQ(s.get_err_msg(), "功能未实现");

    LOG(INFO) << "FunctionNotImplement test passed.";
}

TEST(StatusTest, KeyHasExits) {
    LOG(INFO) << "Testing base::error::KeyHasExits";

    base::Status s = base::error::KeyHasExits("键已存在");
    LOG(INFO) << "Status created: " << s;

    EXPECT_FALSE(s);
    EXPECT_EQ(s.get_err_code(), base::kKeyValueHasExist);
    EXPECT_EQ(s.get_err_msg(), "键已存在");

    LOG(INFO) << "KeyHasExits test passed.";
}

TEST(StatusTest, AssignmentOperator) {
    LOG(INFO) << "Testing assignment operator";

    base::Status s = base::kInvalidArgument;
    LOG(INFO) << "Assigned error code: " << s.get_err_code();

    EXPECT_EQ(s.get_err_code(), base::kInvalidArgument);
    EXPECT_FALSE(s);

    LOG(INFO) << "AssignmentOperator test passed.";
}
