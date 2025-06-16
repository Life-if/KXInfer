//
// Created by asus on 25-6-6.
//
#include <gtest/gtest.h>
#include <glog/logging.h>  // 引入 Google Logging
#include "base/buffer.h"
#include "base/alloc.h"
#include <cuda_runtime_api.h>

using namespace base;

TEST(BufferTest, DefaultConstructor) {
    LOG(INFO) << "开始执行 BufferTest.DefaultConstructor";

    Buffer buffer;

    EXPECT_EQ(buffer.byte_size(), 0);                    // 检查默认构造后大小为0
    EXPECT_EQ(buffer.ptr(), nullptr);                   // 检查指针为空
    EXPECT_EQ(buffer.device_type(), DeviceType::kDeviceUnknown);  // 设备类型未知
    EXPECT_FALSE(buffer.is_external());                 // 默认不是外部内存

    LOG(INFO) << "BufferTest.DefaultConstructor: 测试通过";
}


TEST(BufferTest, ConstructWithSizeAndAllocator) {
    CPUDeviceAllocator cpu_alloc;
    std::shared_ptr<DeviceAllocator> allocator = std::make_shared<CPUDeviceAllocator>();

    Buffer buffer(1024, allocator);

    EXPECT_EQ(buffer.byte_size(), 1024);
    EXPECT_NE(buffer.ptr(), nullptr);
    EXPECT_EQ(buffer.device_type(), DeviceType::kDeviceCPU);
    EXPECT_FALSE(buffer.is_external());

    LOG(INFO) << "BufferTest.ConstructWithSizeAndAllocator: 测试通过";
}

TEST(BufferTest, Allocate) {
    LOG(INFO) << "开始执行 BufferTest.Allocate";
    CPUDeviceAllocator cpu_alloc;
    std::shared_ptr<DeviceAllocator> allocator = std::make_shared<CPUDeviceAllocator>();

    Buffer buffer(512, allocator);

    EXPECT_TRUE(buffer.allocate());
    EXPECT_NE(buffer.ptr(), nullptr);

    LOG(INFO) << "BufferTest.Allocate: 测试通过";
}

TEST(BufferTest, ExternalMemory) {
    LOG(INFO) << "开始执行 BufferTest.ExternalMemory";
    char data[1024] = {};
    CPUDeviceAllocator cpu_alloc;
    std::shared_ptr<DeviceAllocator> allocator = std::make_shared<CPUDeviceAllocator>();

    Buffer buffer(1024, allocator, data, true);

    EXPECT_EQ(buffer.ptr(), data);
    EXPECT_EQ(buffer.byte_size(), 1024);
    EXPECT_TRUE(buffer.is_external());

    LOG(INFO) << "BufferTest.ExternalMemory: 测试通过";
}

TEST(BufferTest, CopyFrom_CPU_To_CPU) {

    LOG(INFO) << "开始执行 BufferTest.CopyFrom_CPU_To_CPU";
    CPUDeviceAllocator cpu_alloc;
    std::shared_ptr<DeviceAllocator> allocator = std::make_shared<CPUDeviceAllocator>();

    Buffer src(32, allocator);
    Buffer dst(32, allocator);

    // 填充源数据
    memset(src.ptr(), 'A', 32);

    // 执行拷贝
    dst.copy_from(src);

    // 验证结果
    for (size_t i = 0; i < dst.byte_size(); ++i) {
        EXPECT_EQ(static_cast<char*>(dst.ptr())[i], 'A');
    }
    LOG(INFO) << "BufferTest.CopyFrom_CPU_To_CPU: 测试通过";
}

TEST(BufferTest, GetDeviceTypeAndSet) {
    LOG(INFO) << "开始执行 BufferTest.GetDeviceTypeAndSet";
    Buffer buffer;

    buffer.set_device_type(DeviceType::kDeviceCUDA);
    EXPECT_EQ(buffer.device_type(), DeviceType::kDeviceCUDA);
    LOG(INFO) << "BufferTest.GetDeviceTypeAndSet: 测试通过";
}

TEST(BufferTest, IsExternal) {
    LOG(INFO) << "开始执行 BufferTest.IsExternal";
    CPUDeviceAllocator cpu_alloc;
    std::shared_ptr<DeviceAllocator> allocator = std::make_shared<CPUDeviceAllocator>();
    char data[1024];

    Buffer external(1024, allocator, data, true);
    Buffer internal(1024, allocator);

    EXPECT_TRUE(external.is_external());
    EXPECT_FALSE(internal.is_external());
    LOG(INFO) << "BufferTest.IsExternal: 测试通过";
}

TEST(BufferTest, CopyFrom_CPU_To_GPU) {
    LOG(INFO) << "开始执行 BufferTest.CopyFrom_CPU_To_GPU";

    // 准备 CPU 分配器和 GPU 分配器
    std::shared_ptr<DeviceAllocator> cpu_alloc = std::make_shared<CPUDeviceAllocator>();
    std::shared_ptr<DeviceAllocator> cuda_alloc = std::make_shared<CUDADeviceAllocator>();

    // 创建源 buffer（CPU）
    Buffer src(32, cpu_alloc);
    // 创建目标 buffer（GPU）
    Buffer dst(32, cuda_alloc);

    // 填充源数据
    memset(src.ptr(), 'A', 32);

    // 执行拷贝
    dst.copy_from(src);

    // 将数据从 GPU 拷贝回 CPU 验证
    char result[32] = {};
    cudaMemcpy(result, dst.ptr(), 32, cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < sizeof(result); ++i) {
        EXPECT_EQ(result[i], 'A');
    }

    LOG(INFO) << "BufferTest.CopyFrom_CPU_To_GPU: 测试通过";
}

TEST(BufferTest, CopyFrom_GPU_To_GPU) {
    LOG(INFO) << "开始执行 BufferTest.CopyFrom_GPU_To_GPU";

    std::shared_ptr<DeviceAllocator> cuda_alloc = std::make_shared<CUDADeviceAllocator>();

    // 创建两个 GPU buffer
    Buffer src(32, cuda_alloc);
    Buffer dst(32, cuda_alloc);

    // 填充源数据
    char data[32] = {};
    memset(data, 'B', 32);
    cudaMemcpy(src.ptr(), data, 32, cudaMemcpyHostToDevice);

    // 执行拷贝
    dst.copy_from(src);

    // 将结果拷贝回 CPU 验证
    char result[32] = {};
    cudaMemcpy(result, dst.ptr(), 32, cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < sizeof(result); ++i) {
        EXPECT_EQ(result[i], 'B');
    }

    LOG(INFO) << "BufferTest.CopyFrom_GPU_To_GPU: 测试通过";
}

TEST(BufferTest, CopyFrom_GPU_To_CPU) {
    LOG(INFO) << "开始执行 BufferTest.CopyFrom_GPU_To_CPU";

    std::shared_ptr<DeviceAllocator> cpu_alloc = std::make_shared<CPUDeviceAllocator>();
    std::shared_ptr<DeviceAllocator> cuda_alloc = std::make_shared<CUDADeviceAllocator>();

    // 创建源 buffer（GPU）
    Buffer src(32, cuda_alloc);
    // 创建目标 buffer（CPU）
    Buffer dst(32, cpu_alloc);

    // 填充 GPU 数据
    char data[32] = {};
    memset(data, 'C', 32);
    cudaMemcpy(src.ptr(), data, 32, cudaMemcpyHostToDevice);

    // 执行拷贝
    dst.copy_from(src);

    // 直接验证 CPU 数据
    for (size_t i = 0; i < dst.byte_size(); ++i) {
        EXPECT_EQ(static_cast<char*>(dst.ptr())[i], 'C');
    }

    LOG(INFO) << "BufferTest.CopyFrom_GPU_To_CPU: 测试通过";
}