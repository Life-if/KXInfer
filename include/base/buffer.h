//
// Created by asus on 25-6-5.
//

#ifndef KXINFER_INCLUDE_BASE_BUFFER_H_
#define KXINFER_INCLUDE_BASE_BUFFER_H_
#include <memory>
#include "base/alloc.h"

namespace base {

    /**
     * @class Buffer
     * @brief 提供统一接口的内存缓冲区封装类，支持设备内存（如 CUDA）与主机内存的管理。
     *
     * @details `Buffer` 是一个不可复制的类，支持通过智能指针（shared_ptr）进行共享所有权管理。
     * 它可以封装由分配器分配的内存，也可以接受外部传入的内存指针。
     * 支持跨设备拷贝操作，并记录当前内存所在的设备类型。
     *
     * @note 该类继承自 `NoCopyable`，禁止拷贝构造和赋值操作，避免浅拷贝导致的资源管理问题。
     *       同时继承 `std::enable_shared_from_this<Buffer>`，允许安全地从对象内部获取 shared_ptr。
     */
    class Buffer : public NoCopyable, std::enable_shared_from_this<Buffer> {

    // public NoCopyable：表示此类对象不可复制（通常通过删除拷贝构造函数和赋值操作符实现）。
    // std::enable_shared_from_this<Buffer>：允许一个 shared_ptr 管理的对象安全地生成另一个
    // 指向自己的 shared_ptr，常用于回调或需要返回 shared_ptr 的场景。

    private:
        size_t byte_size_ = 0; ///< 缓冲区大小（以字节为单位）
        void* ptr_ = nullptr; ///< 指向实际内存的指针
        bool use_external_ = false; ///< 是否使用外部提供的内存（不由本类负责释放）
        DeviceType device_type_ = DeviceType::kDeviceUnknown; ///< 当前缓冲区内存所在设备类型
        std::shared_ptr<DeviceAllocator> allocator_; ///< 内存分配器，用于分配/释放内存

    public:
        /**
         * @brief 默认构造函数，创建一个空的 Buffer 实例。
         *
         * 初始化所有成员变量为默认值：
         * - `byte_size_ = 0`
         * - `ptr_ = nullptr`
         * - `use_external_ = false`
         * - `device_type_ = kDeviceUnknown`
         * - `allocator_ = nullptr`
         */
        explicit Buffer() = default;

        explicit Buffer(size_t byte_size, std::shared_ptr<DeviceAllocator> allocator = nullptr,
                        void* ptr = nullptr, bool use_external = false);

        virtual ~Buffer();

        /**
         * @brief 根据当前配置分配内存。
         *
         * 使用当前设置的分配器和缓冲区大小来申请内存。
         * 如果已经分配过内存，则会先释放原有内存再重新分配。
         *
         * @return 分配成功返回 true，失败返回 false。
         */
        bool allocate();

        /**
         * @brief 从另一个 Buffer 对象中拷贝数据到当前缓冲区。
         *
         * 数据拷贝的方向由源和目标的设备类型决定。
         * 需要确保两个 Buffer 的大小一致。
         *
         * @param buffer 源 Buffer 对象（const 引用）
         */
        void copy_from(const Buffer& buffer) const;

        void copy_from(const Buffer* buffer) const;

        /**
         * @brief 获取当前缓冲区的数据指针（非 const 版本）。
         *
         * 允许修改缓冲区内容。
         *
         * @return 返回指向缓冲区数据的 void 指针。
         */
        void* ptr();

        const void* ptr() const;

        size_t byte_size() const;

        /**
         * @brief 获取当前使用的内存分配器。
         *
         * @return 返回一个指向 DeviceAllocator 的 shared_ptr。
         */
        std::shared_ptr<DeviceAllocator> allocator() const;

        DeviceType device_type() const;

        void set_device_type(DeviceType device_type);

        std::shared_ptr<Buffer> get_shared_from_this();

        bool is_external() const;
    };
}  // namespace base

#endif