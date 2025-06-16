//
// Created by asus on 25-6-5.
//

#ifndef KXINFER_INCLUDE_BASE_ALLOC_H_
#define KXINFER_INCLUDE_BASE_ALLOC_H_
#include <map>
#include <memory>
#include "base.h"

namespace base {
    /**
     * @brief 内存拷贝方向枚举
     *
     * 表示内存复制的方向类型，用于区分不同设备之间的数据传输方式。
     */
    enum class MemcpyKind {
        kMemcpyCPU2CPU = 0,
        kMemcpyCPU2CUDA = 1,
        kMemcpyCUDA2CPU = 2,
        kMemcpyCUDA2CUDA = 3,
    };

    /**
     * @brief 设备内存分配器抽象基类
     *
     * 提供统一接口用于 CPU/CUDA 设备上的内存分配、释放和拷贝操作。
     */
    class DeviceAllocator {
    public:
        // 防止类的构造函数被隐式调用
        /**
         * @brief 构造函数
         * @param device_type 分配器对应的设备类型
         */
        explicit DeviceAllocator(DeviceType device_type) : device_type_(device_type) {
        }

        /**
         * @brief 获取当前分配器所使用的设备类型
         * @return 设备类型
         */
        virtual DeviceType device_type() const { return device_type_; }


        /**
         * @brief 释放已分配的内存
         * @param ptr 要释放的指针
         */
        virtual void release(void *ptr) const = 0;

        /**
         * @brief 分配指定大小的内存
         * @param byte_size 要分配的字节数
         * @return 指向新分配内存的指针
         */
        virtual void *allocate(size_t byte_size) const = 0;

        /**
         * @brief 在源地址和目标地址之间复制内存
         * @param src_ptr 源地址
         * @param dest_ptr 目标地址
         * @param byte_size 复制的字节数
         * @param memcpy_kind 内存复制方向（CPU/CUDA）
         * @param stream CUDA 流对象（可选）
         * @param need_sync 是否需要同步
         */
        virtual void memcpy(const void *src_ptr, void *dest_ptr, size_t byte_size,
                            MemcpyKind memcpy_kind = MemcpyKind::kMemcpyCPU2CPU, void *stream = nullptr,
                            bool need_sync = false) const;
        /**
         * @brief 将内存区域清零
         * @param ptr 要清零的内存指针
         * @param byte_size 清零的字节数
         * @param stream CUDA 流对象（可选）
         * @param need_sync 是否需要同步
         */
        virtual void memset_zero(void *ptr, size_t byte_size, void *stream, bool need_sync = false);

    private:
        DeviceType device_type_ = DeviceType::kDeviceUnknown;
    };

    /**
     * @brief CPU 设备内存分配器
     *
     * 实现基于 CPU 的内存分配与释放逻辑。
     */
    class CPUDeviceAllocator : public DeviceAllocator {
    public:
        explicit CPUDeviceAllocator();

        void *allocate(size_t byte_size) const override;

        // 派生类中重写虚函数
        void release(void *ptr) const override;
    };

    /**
     * @brief CUDA 内存缓冲区描述结构体
     *
     * 用于记录 CUDA 缓冲区的基本信息。
     */
    struct CudaMemoryBuffer {
        void *data;
        size_t byte_size;
        bool busy;

        CudaMemoryBuffer() = default;

        /**
         * @brief 构造函数
         * @param data 数据指针
         * @param byte_size 缓冲区大小
         * @param busy 是否正在被使用
         */
        CudaMemoryBuffer(void *data, size_t byte_size, bool busy)
            : data(data), byte_size(byte_size), busy(busy) {
        }
    };

    /**
     * @brief CUDA 设备内存分配器
     *
     * 支持内存池机制，优化 CUDA 内存分配与释放效率。
     */
    class CUDADeviceAllocator : public DeviceAllocator {
    public:
        explicit CUDADeviceAllocator();

        void *allocate(size_t byte_size) const override;

        void release(void *ptr) const override;

    private:
        mutable std::map<int, size_t> no_busy_cnt_;
        mutable std::map<int, std::vector<CudaMemoryBuffer> > big_buffers_map_;
        mutable std::map<int, std::vector<CudaMemoryBuffer> > cuda_buffers_map_;
    };

    /**
     * @brief CPU 设备分配器工厂类
     *
     * 使用单例模式提供全局唯一的 CPU 内存分配器实例。 实现方式：懒汉式单例 + 工厂方法封装
     */
    class CPUDeviceAllocatorFactory {
    public:
        /**
         * @brief 获取 CPU 分配器的唯一实例
         * @return 全局共享的 CPU 分配器实例
         */
        static std::shared_ptr<CPUDeviceAllocator> get_instance() {
            if (instance == nullptr) {
                // std::make_shared 是 线程安全的
                // 对同一个共享对象的访问由智能指针内部机制管理（引用计数原子操作）
                //     引用计数（use count）
                //     删除器（deleter）
                //     分配器（allocator）等信息
                instance = std::make_shared<CPUDeviceAllocator>();
            }
            return instance;
        }

    private:
        static std::shared_ptr<CPUDeviceAllocator> instance;
    };

    class CUDADeviceAllocatorFactory {
    public:
        static std::shared_ptr<CUDADeviceAllocator> get_instance() {
            if (instance == nullptr) {
                instance = std::make_shared<CUDADeviceAllocator>();
            }
            return instance;
        }

    private:
        static std::shared_ptr<CUDADeviceAllocator> instance;
    };
} // namespace base
#endif  // KXINFER_INCLUDE_BASE_ALLOC_H_
