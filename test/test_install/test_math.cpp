//
// Created by asus on 25-6-4.
//
#include<armadillo>
#include<glog/logging.h>
#include<gtest/gtest.h>

TEST(Math, Armadillo) {
    arma::fmat A = "1,2,3;";
    arma::fmat B = "4,5,6;";
    arma::fmat C = A + B;
    LOG(INFO) << "TEST Armadillo\n";
    CHECK_EQ(C(0), 5);
    CHECK_EQ(C(1), 7);
    CHECK_EQ(C(2), 9);
    LOG(INFO) << "Armadillo 载入成功\n";
}