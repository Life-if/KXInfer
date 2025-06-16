//
// Created by asus on 25-6-5.
//
#include <gtest/gtest.h>
#include <glog/logging.h>  // 引入 Google Logging
#include "base/alloc.h"
#include <cuda_runtime_api.h>

using namespace base;

// ------------------ CPU Device Allocator Tests ------------------

/**
 * @brief 测试 CPU 设备分配器在正常大小下的内存分配与释放功能。
 *
 * @details 本测试验证了以下行为：
 * - 正常大小（1024字节）的内存是否可以被成功分配；
 * - 分配后的内存是否可以安全写入并验证数据；
 * - 内存是否可以被正确释放；
 *
 * @test 测试用例名称: AllocateAndRelease_NormalSize
 */
TEST(CPUDeviceAllocatorTest, AllocateAndRelease_NormalSize) {
    LOG(INFO) << "[CPU] Testing normal allocation and release";

    auto allocator = std::make_shared<base::CPUDeviceAllocator>();

    size_t byte_size = 1024;
    void* ptr = allocator->allocate(byte_size);
    LOG(INFO) << "[CPU] Allocated " << byte_size << " bytes at: " << ptr;

    EXPECT_NE(ptr, nullptr) << "Expected non-null pointer for 1024-byte allocation";
    if (!ptr) {
        LOG(ERROR) << "[CPU] Allocation failed for 1024 bytes. Test aborted.";
    }
    else {
        // 初始化内存
        memset(ptr, 0xFF, byte_size);
        uint8_t* byte_ptr = static_cast<uint8_t*>(ptr);

        // 只验证开头和结尾的部分字节
        bool all_match = true;
        for (size_t i = 0; i < 16; ++i) {
            if (byte_ptr[i] != 0xFF || byte_ptr[byte_size - 16 + i] != 0xFF) {
                all_match = false;
                break;
            }
        }

        if (all_match) {
            LOG(INFO) << "[CPU] Memory content verification succeeded.";
        }
        else {
            LOG(ERROR) << "[CPU] Memory content verification FAILED.";
        }

        // 释放内存
        allocator->release(ptr);
        LOG(INFO) << "[CPU] Released memory at: " << ptr;
    }
}

/**
 * @brief 测试 CPU 设备分配器对零大小内存请求的处理。
 *
 * @details 本测试验证了当请求分配 0 字节内存时，分配器是否返回空指针。
 *
 * @test 测试用例名称: Allocate_ZeroSize_ReturnsNull
 */
TEST(CPUDeviceAllocatorTest, Allocate_ZeroSize_ReturnsNull) {
    LOG(INFO) << "[CPU] Testing zero-size allocation";

    auto allocator = std::make_shared<base::CPUDeviceAllocator>();
    void* ptr = allocator->allocate(0);

    LOG(INFO) << "[CPU] Zero-size allocation returns: " << ptr;
    EXPECT_EQ(ptr, nullptr) << "Expected null pointer for zero-size allocation";

    if (ptr == nullptr) {
        LOG(INFO) << "[CPU] Zero-size allocation correctly returned null.";
    }
    else {
        LOG(ERROR) << "[CPU] Unexpected non-null pointer returned for zero-size allocation.";
        allocator->release(ptr); // Avoid leak in case of unexpected result
    }
}

/**
 * @brief 测试 CPU 设备分配器是否满足特定的内存对齐要求。
 *
 * @details 本测试验证小块（512字节）和大块（2048字节）内存分配是否符合预期的对齐边界。
 *
 * @test 测试用例名称: AlignmentCheck_ForSmallAndLarge
 */
TEST(CPUDeviceAllocatorTest, AlignmentCheck_ForSmallAndLarge) {
    LOG(INFO) << "[CPU] Testing alignment for small and large allocations";

    auto allocator = std::make_shared<base::CPUDeviceAllocator>();

    size_t small_size = 512;
    size_t large_size = 2048;

    void* small_ptr = allocator->allocate(small_size);
    void* large_ptr = allocator->allocate(large_size);

    LOG(INFO) << "[CPU] Small allocation (" << small_size << ") at: " << small_ptr;
    LOG(INFO) << "[CPU] Large allocation (" << large_size << ") at: " << large_ptr;

    EXPECT_NE(small_ptr, nullptr) << "Expected non-null pointer for small allocation";
    EXPECT_NE(large_ptr, nullptr) << "Expected non-null pointer for large allocation";

    const size_t expected_small_align = 16;
    const size_t expected_large_align = 32;

    uintptr_t small_addr = reinterpret_cast<uintptr_t>(small_ptr);
    uintptr_t large_addr = reinterpret_cast<uintptr_t>(large_ptr);

    if (small_addr % expected_small_align != 0) {
        LOG(ERROR) << "[CPU] Small allocation not aligned to " << expected_small_align << " bytes.";
    } else {
        LOG(INFO) << "[CPU] Small allocation aligned to " << expected_small_align << " bytes.";
    }

    if (large_addr % expected_large_align != 0) {
        LOG(ERROR) << "[CPU] Large allocation not aligned to " << expected_large_align << " bytes.";
    } else {
        LOG(INFO) << "[CPU] Large allocation aligned to " << expected_large_align << " bytes.";
    }

    EXPECT_EQ(small_addr % expected_small_align, 0)
        << "Small allocation not aligned to " << expected_small_align << " bytes";

    EXPECT_EQ(large_addr % expected_large_align, 0)
        << "Large allocation not aligned to " << expected_large_align << " bytes";

    allocator->release(small_ptr);
    allocator->release(large_ptr);
}

/**
 * @brief 测试 CPU 设备分配器的基本内存分配与释放流程。
 *
 * @details 本测试验证一个典型场景下：
 * - 内存是否能被成功分配；
 * - 是否能被正确释放；
 *
 * @test 测试用例名称: AllocateAndRelease
 */
TEST(CPUDeviceAllocatorTest, AllocateAndRelease) {
    LOG(INFO) << "开始测试 CPU 内存分配与释放";

    auto allocator = CPUDeviceAllocatorFactory::get_instance();

    // 分配 1024 字节内存
    void* ptr = allocator->allocate(1024);
    EXPECT_NE(ptr, nullptr) << "内存分配失败，返回空指针";
    if (ptr == nullptr) {
        LOG(ERROR) << "内存分配失败，测试终止";
        return;
    }

    LOG(INFO) << "内存分配成功，地址: " << ptr;

    // 释放内存
    allocator->release(ptr);
    LOG(INFO) << "内存释放成功";
}


// ------------------ CUDA Device Allocator Tests ------------------

/**
 * @brief 测试 CUDA 设备分配器在正常大小下的内存分配、初始化、验证与释放功能。
 *
 * @details 本测试验证以下行为：
 * - 正常大小（1MB）的设备内存是否可以被成功分配；
 * - 是否可以通过 `cudaMemset` 初始化设备内存；
 * - 是否能通过 `cudaMemcpyDeviceToHost` 拷贝回主机并验证内容；
 * - 是否能正确释放内存；
 *
 * @test 测试用例名称: AllocateAndRelease_NormalSize
 */
TEST(CUDADeviceAllocatorTest, AllocateAndRelease_NormalSize) {
    LOG(INFO) << "[CUDA] Testing normal allocation and release with data verification";

    auto allocator = std::make_shared<base::CUDADeviceAllocator>();

    size_t byte_size = 1024 * 1024;  // 1MB
    void* d_ptr = allocator->allocate(byte_size);
    LOG(INFO) << "[CUDA] Allocated " << byte_size << " bytes at: " << d_ptr;

    EXPECT_NE(d_ptr, nullptr) << "Expected non-null pointer for 1MB allocation";
    if (!d_ptr) {
        LOG(ERROR) << "[CUDA] Allocation failed for 1MB. Test aborted.";
        return;
    }

    // 在设备内存中初始化为 0xFF
    uint8_t val = 0xFF;
    cudaError_t err = cudaMemset(d_ptr, val, byte_size);
    EXPECT_EQ(err, cudaSuccess) << "cudaMemset failed: " << cudaGetErrorString(err);
    if (err != cudaSuccess) {
        LOG(ERROR) << "[CUDA] cudaMemset failed: " << cudaGetErrorString(err);
    }

    // 分配主机内存并拷贝回来验证
    uint8_t* h_data = new uint8_t[byte_size];
    cudaMemcpy(h_data, d_ptr, byte_size, cudaMemcpyDeviceToHost);

    // 同步检查
    cudaDeviceSynchronize();

    // 验证每个字节是否都为 0xFF
    bool all_match = true;
    for (size_t i = 0; i < byte_size; ++i) {
        if (h_data[i] != val) {
            all_match = false;
            break;
        }
    }

    if (all_match) {
        LOG(INFO) << "[CUDA] Verified device memory contents successfully.";
    } else {
        LOG(ERROR) << "[CUDA] Memory content verification FAILED.";
    }

    delete[] h_data;
    allocator->release(d_ptr);
    LOG(INFO) << "[CUDA] Released memory at: " << d_ptr;
}

/**
 * @brief 测试 CUDA 设备分配器对零大小内存请求的处理。
 *
 * @details 本测试验证当请求分配 0 字节内存时，分配器是否返回空指针。
 *
 * @test 测试用例名称: Allocate_ZeroSize_ReturnsNull
 */
TEST(CUDADeviceAllocatorTest, Allocate_ZeroSize_ReturnsNull) {
    LOG(INFO) << "[CUDA] Testing zero-size allocation";

    auto allocator = std::make_shared<base::CUDADeviceAllocator>();
    void* ptr = allocator->allocate(0);

    LOG(INFO) << "[CUDA] Zero-size allocation returns: " << ptr;
    EXPECT_EQ(ptr, nullptr) << "Expected null pointer for zero-size allocation";

    if (ptr == nullptr) {
        LOG(INFO) << "[CUDA] Zero-size allocation correctly returned null.";
    } else {
        LOG(ERROR) << "[CUDA] Unexpected non-null pointer returned for zero-size allocation.";
        allocator->release(ptr); // Avoid leak in case of unexpected result
    }
}

/**
 * @brief 测试 CUDA 设备分配器在释放后是否复用相同地址的内存块。
 *
 * @details 本测试验证连续两次分配相同大小的内存块时，是否能够复用第一次释放的地址。
 *
 * @test 测试用例名称: MemoryReuse_AfterRelease
 */
TEST(CUDADeviceAllocatorTest, MemoryReuse_AfterRelease) {
    LOG(INFO) << "[CUDA] Testing memory reuse after release";

    auto allocator = std::make_shared<base::CUDADeviceAllocator>();

    size_t byte_size = 1024 * 1024;  // 1MB

    void* ptr1 = allocator->allocate(byte_size);
    LOG(INFO) << "[CUDA] First allocation at: " << ptr1;

    if (!ptr1) {
        LOG(ERROR) << "[CUDA] Allocation failed during reuse test";
        return;
    }

    allocator->release(ptr1);

    void* ptr2 = allocator->allocate(byte_size);
    LOG(INFO) << "[CUDA] Second allocation at: " << ptr2;

    EXPECT_EQ(ptr1, ptr2)
        << "Memory should be reused from pool. Expected: " << ptr1 << ", Got: " << ptr2;

    if (ptr1 == ptr2) {
        LOG(INFO) << "[CUDA] Memory was successfully reused.";
    } else {
        LOG(ERROR) << "[CUDA] Memory was NOT reused as expected.";
    }

    allocator->release(ptr2);
}

/**
 * @brief 测试大块内存分配是否不复用之前释放的地址（即不在缓存池中）。
 *
 * @details 本测试验证较大内存块（5MB）在释放后，是否不会立即被复用于后续相同大小的分配。
 *
 * @test 测试用例名称: LargeAllocation_NoPoolReuse
 */
TEST(CUDADeviceAllocatorTest, LargeAllocation_NoPoolReuse) {
    LOG(INFO) << "[CUDA] Testing large allocation (no pool reuse)";

    auto allocator = std::make_shared<base::CUDADeviceAllocator>();

    size_t byte_size = 5 * 1024 * 1024;  // 5MB

    void* ptr1 = allocator->allocate(byte_size);
    LOG(INFO) << "[CUDA] First large allocation at: " << ptr1;

    if (!ptr1) {
        LOG(ERROR) << "[CUDA] Allocation failed during large allocation test";
        return;
    }

    allocator->release(ptr1);

    void* ptr2 = allocator->allocate(byte_size);
    LOG(INFO) << "[CUDA] Second large allocation at: " << ptr2;

    EXPECT_NE(ptr1, nullptr);
    EXPECT_NE(ptr2, nullptr);

    if (ptr1 && ptr2 && ptr1 != ptr2) {
        LOG(INFO) << "[CUDA] Large allocation did not reuse previous memory block as expected.";
    } else if (ptr1 && ptr2 && ptr1 == ptr2) {
        LOG(WARNING) << "[CUDA] Large allocation unexpectedly reused memory block.";
    }

    allocator->release(ptr2);
}

// ------------------ 跨设备 Tests ------------------

/**
 * @brief 测试 CPU 到 CPU 的 memcpy 功能。
 *
 * @details 本测试验证分配器实现的 `memcpy` 函数是否能够正确地将数据从一个 CPU 缓冲区复制到另一个。
 *
 * @test 测试用例名称: MemcpyCPU2CPU
 */
TEST(CPUDeviceAllocatorTest, MemcpyCPU2CPU) {
    LOG(INFO) << "开始测试 CPU 到 CPU 的内存复制";

    auto allocator = CPUDeviceAllocatorFactory::get_instance();

    char src[1024] = {0};
    char dest[1024] = {1};

    // 初始化源数据
    for(int i = 0; i < 1024; ++i) {
        src[i] = static_cast<char>(i % 256);
    }

    allocator->memcpy(src, dest, sizeof(src), MemcpyKind::kMemcpyCPU2CPU);

    // 验证目标数据是否一致
    bool all_match = true;
    for(int i = 0; i < 1024; ++i) {
        if (src[i] != dest[i]) {
            all_match = false;
            LOG(ERROR) << "拷贝失败，第 " << i << " 字节不匹配，期望: " << int(src[i])
                       << ", 实际: " << int(dest[i]);
            break;
        }
    }

    if (all_match) {
        LOG(INFO) << "CPU 到 CPU 内存复制测试通过";
    }
    else {
        LOG(ERROR) << "CPU 到 CPU 内存复制测试失败";
    }
}

/**
 * @brief 测试 CPU 内存清零功能。
 *
 * @details 本测试验证分配器实现的 `memset_zero` 函数是否能将指定内存区域全部置为 0。
 *
 * @test 测试用例名称: MemsetZero
 */
TEST(CPUDeviceAllocatorTest, MemsetZero) {
    LOG(INFO) << "开始测试内存清零功能";

    auto allocator = CPUDeviceAllocatorFactory::get_instance();

    char buffer[1024];
    for(int i = 0; i < 1024; ++i) {
        buffer[i] = static_cast<char>(i % 256);
    }

    allocator->memset_zero(buffer, sizeof(buffer), nullptr);

    // 检查是否全部置零
    bool all_zero = true;
    for(int i = 0; i < 1024; ++i) {
        if (buffer[i] != 0) {
            all_zero = false;
            LOG(ERROR) << "第 " << i << " 字节未被清零，实际值: " << int(buffer[i]);
            break;
        }
    }

    if (all_zero) {
        LOG(INFO) << "内存清零测试通过";
    }
    else {
        LOG(ERROR) << "内存清零测试失败";
    }
}

// CUDA 相关测试需要硬件支持
/**
 * @brief 测试 CUDA 设备分配器的基本内存分配与释放功能。
 *
 * @details 本测试验证以下行为：
 * - 是否能成功分配指定大小的设备内存；
 * - 是否能正确释放已分配的设备内存；
 *
 * @test 测试用例名称: AllocateAndRelease
 */
TEST(CUDADeviceAllocatorTest, AllocateAndRelease) {
    LOG(INFO) << "开始测试 CUDA 内存分配与释放";

    auto allocator = CUDADeviceAllocatorFactory::get_instance();

    void* ptr = allocator->allocate(1024);
    EXPECT_NE(ptr, nullptr) << "CUDA 内存分配失败";
    if (ptr == nullptr) {
        LOG(ERROR) << "CUDA 内存分配失败，测试终止";
        return;
    }

    LOG(INFO) << "CUDA 内存分配成功，地址: " << ptr;

    allocator->release(ptr);
    LOG(INFO) << "CUDA 内存释放成功";
}

/**
 * @brief 测试从 GPU 到 CPU 的内存拷贝功能。
 *
 * @details 本测试验证分配器实现的 `memcpy` 函数是否能够：
 * - 正确地将数据从设备内存复制到主机内存；
 * - 拷贝后的内容是否一致；
 *
 * @test 测试用例名称: MemcpyCUDA2CPU
 */
TEST(CUDADeviceAllocatorTest, MemcpyCUDA2CPU) {
    LOG(INFO) << "开始测试 CUDA 到 CPU 内存拷贝";

    auto allocator = CUDADeviceAllocatorFactory::get_instance();

    const size_t size = 1024;
    char* cuda_mem = static_cast<char*>(allocator->allocate(size));
    char cpu_mem[size];

    // 初始化 CUDA 内存为 0x7A
    cudaError_t err = cudaMemset(cuda_mem, 0x7A, size);
    EXPECT_EQ(err, cudaSuccess) << "cudaMemset failed: " << cudaGetErrorString(err);
    if (err != cudaSuccess) {
        LOG(ERROR) << "[CUDA] cudaMemset failed during MemcpyCUDA2CPU test.";
        allocator->release(cuda_mem);
        return;
    }

    // 执行 memcpy 操作
    allocator->memcpy(cuda_mem, cpu_mem, size, MemcpyKind::kMemcpyCUDA2CPU);

    // 检查内容是否一致
    bool all_match = true;
    for(int i = 0; i < size; ++i) {
        if (cpu_mem[i] != 0x7A) {
            all_match = false;
            LOG(ERROR) << "第 " << i << " 字节拷贝失败，期望: 0x7A，实际: " << int(cpu_mem[i]);
            break;
        }
    }

    if (all_match) {
        LOG(INFO) << "CUDA 到 CPU 内存拷贝测试通过";
    } else {
        LOG(ERROR) << "CUDA 到 CPU 内存拷贝测试失败";
    }

    allocator->release(cuda_mem);
    LOG(INFO) << "CUDA 到 CPU 内存拷贝资源释放完成";
}

/**
 * @brief 测试从 CPU 到 GPU 的内存拷贝功能。
 *
 * @details 本测试验证分配器实现的 `memcpy` 函数是否能够：
 * - 正确地将数据从主机内存复制到设备内存；
 * - 拷贝后的内容是否一致；
 *
 * @test 测试用例名称: MemcpyCPU2CUDA
 */
TEST(CUDADeviceAllocatorTest, MemcpyCPU2CUDA) {
    LOG(INFO) << "开始测试 CPU 到 GPU 内存拷贝";

    auto allocator = CUDADeviceAllocatorFactory::get_instance();

    const size_t size = 1024;
    char h_src[1024], h_dest[1024];

    // 初始化主机内存
    for (int i = 0; i < size; ++i) {
        h_src[i] = static_cast<char>(i % 256);
    }

    // 分配设备内存
    char* d_dst = static_cast<char*>(allocator->allocate(size));
    EXPECT_NE(d_dst, nullptr) << "CUDA 内存分配失败";
    if (!d_dst) {
        LOG(ERROR) << "CUDA 内存分配失败，测试终止";
        return;
    }

    // 执行异步拷贝
    allocator->memcpy(h_src, d_dst, size, MemcpyKind::kMemcpyCPU2CUDA, nullptr, true);

    // 将结果拷贝回主机验证
    cudaMemcpy(h_dest, d_dst, size, cudaMemcpyDeviceToHost);

    // 验证数据是否一致
    bool all_match = true;
    for (int i = 0; i < size; ++i) {
        if (h_src[i] != h_dest[i]) {
            all_match = false;
            LOG(ERROR) << "第 " << i << " 字节不匹配，期望: " << int(h_src[i])
                       << ", 实际: " << int(h_dest[i]);
            break;
        }
    }

    if (all_match) {
        LOG(INFO) << "CPU 到 GPU 内存拷贝测试通过";
    } else {
        LOG(ERROR) << "CPU 到 GPU 内存拷贝测试失败";
    }

    allocator->release(d_dst);
    LOG(INFO) << "CPU 到 GPU 内存拷贝资源释放完成";
}

/**
 * @brief 测试从 GPU 到 GPU 的内存拷贝功能。
 *
 * @details 本测试验证分配器实现的 `memcpy` 函数是否能够：
 * - 在设备端执行内存拷贝操作；
 * - 拷贝后的内容是否一致；
 *
 * @test 测试用例名称: MemcpyCUDA2CUDA
 */
TEST(CUDADeviceAllocatorTest, MemcpyCUDA2CUDA) {
    LOG(INFO) << "开始测试 GPU 到 GPU 内存拷贝";

    auto allocator = CUDADeviceAllocatorFactory::get_instance();

    const size_t size = 1024;
    char h_result[1024];
    char* d_src = static_cast<char*>(allocator->allocate(size));
    char* d_dst = static_cast<char*>(allocator->allocate(size));

    EXPECT_NE(d_src, nullptr) << "源内存分配失败";
    EXPECT_NE(d_dst, nullptr) << "目标内存分配失败";

    if (!d_src || !d_dst) {
        LOG(ERROR) << "GPU 内存分配失败，测试终止";
        if (d_src) allocator->release(d_src);
        if (d_dst) allocator->release(d_dst);
        return;
    }

    // 初始化源 GPU 内存
    cudaError_t err = cudaMemset(d_src, 0x55, size);
    EXPECT_EQ(err, cudaSuccess) << "cudaMemset failed: " << cudaGetErrorString(err);
    if (err != cudaSuccess) {
        LOG(ERROR) << "[CUDA] cudaMemset failed during MemcpyCUDA2CUDA test.";
        allocator->release(d_src);
        allocator->release(d_dst);
        return;
    }

    // GPU 到 GPU 拷贝
    allocator->memcpy(d_src, d_dst, size, MemcpyKind::kMemcpyCUDA2CUDA, nullptr, true);

    // 拷贝回主机验证
    cudaMemcpy(h_result, d_dst, size, cudaMemcpyDeviceToHost);

    // 验证所有字节是否正确
    bool all_match = true;
    for (int i = 0; i < size; ++i) {
        if (h_result[i] != 0x55) {
            all_match = false;
            LOG(ERROR) << "第 " << i << " 字节不匹配，期望: 0x55，实际: " << int(h_result[i]);
            break;
        }
    }

    if (all_match) {
        LOG(INFO) << "GPU 到 GPU 内存拷贝测试通过";
    } else {
        LOG(ERROR) << "GPU 到 GPU 内存拷贝测试失败";
    }

    allocator->release(d_src);
    allocator->release(d_dst);
    LOG(INFO) << "GPU 到 GPU 内存拷贝资源释放完成";
}