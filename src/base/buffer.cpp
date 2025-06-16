#include "base/buffer.h"
#include <glog/logging.h>

namespace base {
    /**
     * @brief 构造一个 Buffer 对象。
     *
     * @param byte_size 缓冲区大小（字节）
     * @param allocator 内存分配器，默认为 nullptr
     * @param ptr 外部传入的内存指针，若非空则使用该内存
     * @param use_external 是否使用外部内存（true 表示不负责释放内存）
     */
    Buffer::Buffer(size_t byte_size,
                   std::shared_ptr<DeviceAllocator> allocator,
                   void *ptr,
                   bool use_external)
        : byte_size_(byte_size),
          allocator_(allocator),
          ptr_(ptr),
          use_external_(use_external) {
        if (!ptr_ && allocator_) {
            device_type_ = allocator_->device_type();
            use_external_ = false;
            ptr_ = allocator_->allocate(byte_size);
        }
    }

    /**
     * @brief 析构函数，释放缓冲区内存（除非使用的是外部内存）。
     */
    Buffer::~Buffer() {
        if (!use_external_) {
            if (ptr_ && allocator_) {
                allocator_->release(ptr_);
                ptr_ = nullptr;
            }
        }
    }


    void *Buffer::ptr() {
        return ptr_;
    }

    const void *Buffer::ptr() const {
        return ptr_;
    }

    size_t Buffer::byte_size() const {
        return byte_size_;
    }

    /**
     * @brief 根据当前配置分配内存。
     *
     * 使用当前设置的分配器和缓冲区大小来申请内存。
     * 如果已经分配过内存，则会先释放原有内存再重新分配。
     *
     * @return 分配成功返回 true，失败返回 false。
     */
    bool Buffer::allocate() {
        if (allocator_ && byte_size_ != 0) {
            use_external_ = false;
            ptr_ = allocator_->allocate(byte_size_);
            if (!ptr_) {
                return false;
            }
            else {
                return true;
            }
        }
        else {
            return false;
        }
    }

    std::shared_ptr<DeviceAllocator> Buffer::allocator() const {
        return allocator_;
    }

    /**
     * @brief 从另一个 Buffer 对象中拷贝数据到当前缓冲区。
     *
     * 数据拷贝的方向由源和目标的设备类型决定。
     * 需要确保两个 Buffer 的大小一致。
     *
     * @param buffer 源 Buffer 对象（const 引用）
     */
    void Buffer::copy_from(const Buffer &buffer) const {
        CHECK(allocator_ != nullptr);
        CHECK(buffer.ptr_ != nullptr);

        // 确定实际要拷贝的字节数为两个 Buffer 中较小的那个。
        size_t byte_size = byte_size_ < buffer.byte_size_ ? byte_size_ : buffer.byte_size_;

        // 获取源 Buffer 和当前 Buffer 所在的设备类型（如 CPU 或 CUDA GPU）
        const DeviceType &buffer_device = buffer.device_type();
        const DeviceType &current_device = this->device_type();

        CHECK(buffer_device != DeviceType::kDeviceUnknown &&
            current_device != DeviceType::kDeviceUnknown);

        if (buffer_device == DeviceType::kDeviceCPU &&
            current_device == DeviceType::kDeviceCPU) {
            return allocator_->memcpy(buffer.ptr(), this->ptr_, byte_size);
        }
        else if (buffer_device == DeviceType::kDeviceCUDA &&
                   current_device == DeviceType::kDeviceCPU) {
            return allocator_->memcpy(buffer.ptr(), this->ptr_, byte_size,
                                      MemcpyKind::kMemcpyCUDA2CPU);
        }
        else if (buffer_device == DeviceType::kDeviceCPU &&
                   current_device == DeviceType::kDeviceCUDA) {
            return allocator_->memcpy(buffer.ptr(), this->ptr_, byte_size,
                                      MemcpyKind::kMemcpyCPU2CUDA);
        }
        else {
            return allocator_->memcpy(buffer.ptr(), this->ptr_, byte_size,
                                      MemcpyKind::kMemcpyCUDA2CUDA);
        }
    }

    void Buffer::copy_from(const Buffer *buffer) const {
        CHECK(allocator_ != nullptr);
        CHECK(buffer != nullptr || buffer->ptr_ != nullptr);

        size_t dest_size = byte_size_;
        size_t src_size = buffer->byte_size_;
        size_t byte_size = src_size < dest_size ? src_size : dest_size;

        const DeviceType &buffer_device = buffer->device_type();
        const DeviceType &current_device = this->device_type();
        CHECK(buffer_device != DeviceType::kDeviceUnknown &&
            current_device != DeviceType::kDeviceUnknown);

        if (buffer_device == DeviceType::kDeviceCPU &&
            current_device == DeviceType::kDeviceCPU) {
            return allocator_->memcpy(buffer->ptr_, this->ptr_, byte_size);
        }
        else if (buffer_device == DeviceType::kDeviceCUDA &&
                   current_device == DeviceType::kDeviceCPU) {
            return allocator_->memcpy(buffer->ptr_, this->ptr_, byte_size,
                                      MemcpyKind::kMemcpyCUDA2CPU);
        }
        else if (buffer_device == DeviceType::kDeviceCPU &&
                   current_device == DeviceType::kDeviceCUDA) {
            return allocator_->memcpy(buffer->ptr_, this->ptr_, byte_size,
                                      MemcpyKind::kMemcpyCPU2CUDA);
        }
        else {
            return allocator_->memcpy(buffer->ptr_, this->ptr_, byte_size,
                                      MemcpyKind::kMemcpyCUDA2CUDA);
        }
    }

    DeviceType Buffer::device_type() const {
        return device_type_;
    }

    void Buffer::set_device_type(DeviceType device_type) {
        device_type_ = device_type;
    }

    std::shared_ptr<Buffer> Buffer::get_shared_from_this() {
        return shared_from_this();
    }

    bool Buffer::is_external() const {
        return this->use_external_;
    }
} // namespace base
