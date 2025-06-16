//
// Created by asus on 25-6-5.
//

/**
 * @file alloc.cpp
 * @brief 设备内存分配器核心实现文件
 *
 * 提供跨设备（CPU/CUDA）的内存复制与清零功能。
 */

#include "base/alloc.h"
#include <cstdlib>
#include <cuda_runtime_api.h>

namespace base {
    void DeviceAllocator::memcpy(const void *src_ptr,
                                 void *dest_ptr,
                                 size_t byte_size,
                                 MemcpyKind memcpy_kind,
                                 void *stream,
                                 bool need_sync) const {
        CHECK_NE(src_ptr, nullptr);
        CHECK_NE(dest_ptr, nullptr);
        if (!byte_size) {
            return;
        }

        cudaStream_t stream_ = nullptr;
        if (stream) {
            stream_ = static_cast<CUstream_st *>(stream);
        }

        if (memcpy_kind == MemcpyKind::kMemcpyCPU2CPU) {
            std::memcpy(dest_ptr, src_ptr, byte_size);
        } else if (memcpy_kind == MemcpyKind::kMemcpyCPU2CUDA) {
            if (!stream_) {
                cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyHostToDevice);
            } else {
                cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyHostToDevice, stream_);
            }
        } else if (memcpy_kind == MemcpyKind::kMemcpyCUDA2CPU) {
            if (!stream_) {
                cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToHost);
            } else {
                cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToHost, stream_);
            }
        } else if (memcpy_kind == MemcpyKind::kMemcpyCUDA2CUDA) {
            if (!stream_) {
                cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToDevice);
            } else {
                cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyDeviceToDevice, stream_);
            }
        } else {
            LOG(FATAL) << "Unknown memcpy kind: " << int(memcpy_kind);
        }

        // 如果设置了 need_sync，则等待所有异步操作完成。
        if (need_sync) {
            cudaDeviceSynchronize();
        }
    }

    void DeviceAllocator::memset_zero(void *ptr,
                                      size_t byte_size,
                                      void *stream,
                                      bool need_sync) {
        CHECK(device_type_ != base::DeviceType::kDeviceUnknown);
        if (device_type_ == base::DeviceType::kDeviceCPU) {
            std::memset(ptr, 0, byte_size);
        }
        else {
            if (stream) {
                cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
                cudaMemsetAsync(ptr, 0, byte_size, stream_);
            }
            else {
                cudaMemset(ptr, 0, byte_size);
            }
            if (need_sync) {
                cudaDeviceSynchronize();
            }
        }
    }
} // namespace base


////////////////////////////////// CUDA ///////////////////////////////////////////

namespace base {
    CUDADeviceAllocator::CUDADeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCUDA) {
    }

    /**
     * @brief 分配指定大小的 GPU 内存
     *
     * 根据请求大小选择合适的内存块：
     * - 如果大于 1MB：尝试复用大内存块或申请新内存
     * - 如果小于等于 1MB：尝试复用小内存块或申请新内存
     * 内存池机制旨在减少频繁调用 cudaMalloc/cudaFree 带来的性能损耗。
     *
     * @param byte_size 请求的字节数
     * @return 成功返回指针，失败返回 nullptr
     */
    void *CUDADeviceAllocator::allocate(size_t byte_size) const {
        // 获取当前 CUDA 设备 ID；
        int id = -1;
        cudaError_t state = cudaGetDevice(&id);
        CHECK(state == cudaSuccess);

        // 大于 1MB 的内存请求走 big_buffers_map_ 流程
        if (byte_size > 1024 * 1024) {
            auto &big_buffers = big_buffers_map_[id];
            int sel_id = -1;    // 用于记录选中的缓冲区索引

            // 遍历所有大内存块，寻找最优匹配
            for (int i = 0; i < big_buffers.size(); i++) {
                if (big_buffers[i].byte_size >= byte_size && !big_buffers[i].busy &&
                    big_buffers[i].byte_size - byte_size < 1 * 1024 * 1024) {
                    // 若是第一个满足条件的或更优解（容量更小），更新 sel_id
                    if (sel_id == -1 || big_buffers[sel_id].byte_size > big_buffers[i].byte_size) {
                        sel_id = i;
                    }
                }
            }

            // 如果找到合适的内存块
            if (sel_id != -1) {
                big_buffers[sel_id].busy = true;
                return big_buffers[sel_id].data;
            }

            // 没有合适的缓存块，直接申请新内存
            void *ptr = nullptr;
            state = cudaMalloc(&ptr, byte_size);    // 实际调用 CUDA 分配函数
            if (cudaSuccess != state) {
                char buf[256];
                snprintf(buf, 256,
                         "Error: CUDA error when allocating %lu MB memory! maybe there's no enough memory "
                         "left on  device.",
                         byte_size >> 20);
                LOG(ERROR) << buf;
                return nullptr;
            }
            // 成功分配后加入内存池
            big_buffers.emplace_back(ptr, byte_size, true);
            return ptr;
        }

        // 小于等于 1MB 的内存请求走 cuda_buffers_map_ 流程
        // 上面的可能性直接返回指针了
        auto &cuda_buffers = cuda_buffers_map_[id];
        for (int i = 0; i < cuda_buffers.size(); i++) {
            if (cuda_buffers[i].byte_size >= byte_size && !cuda_buffers[i].busy) {
                cuda_buffers[i].busy = true;
                no_busy_cnt_[id] -= cuda_buffers[i].byte_size;    // 更新空闲内存计数
                return cuda_buffers[i].data;
            }
        }

        // 没有可用缓存，申请新内存 小内存
        void *ptr = nullptr;
        state = cudaMalloc(&ptr, byte_size);
        if (cudaSuccess != state) {
            char buf[256];
            snprintf(buf, 256,
                     "Error: CUDA error when allocating %lu MB memory! maybe there's no enough memory "
                     "left on  device.",
                     byte_size >> 20);
            LOG(ERROR) << buf;
            return nullptr;
        }
        cuda_buffers.emplace_back(ptr, byte_size, true);
        return ptr;
    }


    /**
     * @brief 释放之前分配的 GPU 内存
     *
     * 该方法会尝试将内存回收到内存池中以供后续复用。
     * 如果内存不在内存池中，则直接调用 cudaFree 进行释放。
     *
     * @param[in] ptr 要释放的内存指针
     */
    void CUDADeviceAllocator::release(void *ptr) const {
        if (!ptr) {
            return;
        }
        if (cuda_buffers_map_.empty()) {
            return;
        }

        cudaError_t state = cudaSuccess;

        // 清理长时间空闲的内存块（总空闲超过 1GB）
        for (auto &it: cuda_buffers_map_) {
            if (no_busy_cnt_[it.first] > 1024 * 1024 * 1024) {
                auto &cuda_buffers = it.second;
                std::vector<CudaMemoryBuffer> temp;
                for (int i = 0; i < cuda_buffers.size(); i++) {
                    if (!cuda_buffers[i].busy) {
                        state = cudaSetDevice(it.first);    // 切换到目标设备
                        state = cudaFree(cuda_buffers[i].data);
                        CHECK(state == cudaSuccess)
              << "Error: CUDA error when release memory on device " << it.first;
                    }
                    else {
                        temp.push_back(cuda_buffers[i]);    // 保留正在使用的块
                    }
                }
                cuda_buffers.clear();
                it.second = temp;
                no_busy_cnt_[it.first] = 0;
            }
        }

        // 尝试将内存回收回内存池
        for (auto &it: cuda_buffers_map_) {
            auto &cuda_buffers = it.second;
            for (int i = 0; i < cuda_buffers.size(); i++) {
                if (cuda_buffers[i].data == ptr) {
                    no_busy_cnt_[it.first] += cuda_buffers[i].byte_size;
                    cuda_buffers[i].busy = false;
                    return;
                }
            }
            auto &big_buffers = big_buffers_map_[it.first];
            for (int i = 0; i < big_buffers.size(); i++) {
                if (big_buffers[i].data == ptr) {
                    big_buffers[i].busy = false;
                    return;
                }
            }
        }
        // 如果没有在内存池中找到该指针，直接释放
        state = cudaFree(ptr);
        CHECK(state == cudaSuccess) << "Error: CUDA error when release memory on device";
    }

    std::shared_ptr<CUDADeviceAllocator> CUDADeviceAllocatorFactory::instance = nullptr;
} // namespace base


//////////////////////// CPU ////////////////////////////////////

#if (defined(_POSIX_ADVISORY_INFO) && (_POSIX_ADVISORY_INFO >= 200112L))
#define KXINFER_HAVE_POSIX_MEMALIGN
#endif

namespace base {
    CPUDeviceAllocator::CPUDeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCPU) {
    }

    /**
     * @brief 分配指定大小的 CPU 内存
     *
     * 支持按需对齐：
     * - 如果请求大小 >= 1024 字节，使用 32 字节对齐；
     * - 否则使用 16 字节对齐；
     * - 若平台不支持 posix_memalign，则回退到 malloc。
     *
     * @param[in] byte_size 请求分配的字节数
     * @return 成功时返回指向 CPU 内存的指针，失败返回 nullptr
     */
    void* CPUDeviceAllocator::allocate(size_t byte_size) const {
        if (!byte_size) {
            return nullptr;
        }
#ifdef KXINFER_HAVE_POSIX_MEMALIGN
        void* data = nullptr;
        const size_t alignment = (byte_size >= size_t(1024)) ? size_t(32) : size_t(16);
        int status = posix_memalign((void**)&data,
                                    ((alignment >= sizeof(void*)) ? alignment : sizeof(void*)),
                                    byte_size);
        if (status != 0) {
            return nullptr;
        }
        return data;
#else
        void* data = malloc(byte_size);
        return data;
#endif
    }

    void CPUDeviceAllocator::release(void* ptr) const {
        if (ptr) {
            free(ptr);
        }
    }

    std::shared_ptr<CPUDeviceAllocator> CPUDeviceAllocatorFactory::instance = nullptr;
}  // namespace base