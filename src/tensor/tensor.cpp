//
// Created by asus on 25-6-8.
//

#include "tensor/tensor.h"
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <numeric>     // 用于维度相乘。

namespace tensor {
    // 仅在类内部使用，限制作用域和访问权限
    // 避免污染全局命名空间

    /**
     * @brief 计算指定维度范围内所有元素的乘积，用于计算张量的数据总量。
     *
     * @tparam T  迭代器类型，用于访问维度数组。
     * @tparam Tp 初始值类型，通常为 size_t。
     * @param begin 指向维度数组起始位置的迭代器。
     * @param end   指向维度数组结束位置的迭代器。
     * @param init  初始乘积值，一般设置为 1。
     * @return 返回维度乘积结果，表示该维度范围内的数据总量。
     */
    template<typename T, typename Tp>
    static size_t reduce_dimension(T begin, T end, Tp init) {
        // 如果 begin >= end，则维度范围无效，返回 0 表示没有数据。
        if (begin >= end) {
            return 0;
        }
        // 使用 std::accumulate 结合 std::multiplies<> 计算从 begin 到 end 的所有维度的乘积。
        // 初始值为 init（通常是 1），从而得到总的元素数量。
        size_t size = std::accumulate(begin, end, init, std::multiplies<>());
        // 返回计算出的维度乘积结果。
        return size;
    }


    /**
     * @brief 获取指定数据类型占用的字节数。
     *
     * @param data_type 数据类型，支持浮点型、8位整型和32位整型。
     * @return 返回对应数据类型的字节大小。未知类型会记录错误日志并返回 0。
     */
    static size_t data_type_size(base::DataType data_type) {
        switch (data_type) {
            // 浮点型数据，占用 4 字节。
            case base::DataType::kDataTypeFp32: {
                return 4;
            }
            // 8 位整型数据，占用 1 字节。
            case base::DataType::kDataTypeInt8: {
                return 1;
            }
            // 32 位整型数据，占用 4 字节。
            case base::DataType::kDataTypeInt32: {
                return 4;
            }
            // 默认情况：遇到不支持的数据类型时，记录致命错误日志并返回 0。
            default: {
                LOG(FATAL) << "Unknown data type size for " << int(data_type);
                return 0;
            }
        }
    }


    /**
     * @brief 构造一个一维张量对象。
     *
     * @param data_type 张量中元素的数据类型。
     * @param dim0 第一个维度的大小。
     * @param need_alloc 是否需要分配新内存。
     * @param alloc 用于内存分配的分配器指针。
     * @param ptr 可选的外部数据指针，若提供则使用该指针初始化缓冲区。
     */
    Tensor::Tensor(base::DataType data_type,
                   int32_t dim0,
                   bool need_alloc,
                   std::shared_ptr<base::DeviceAllocator> alloc,
                   void *ptr)
        : data_type_(data_type) {
        // 将输入的维度添加到 dims_ 向量中。
        dims_.push_back(dim0);
        // 设置张量的总大小为第一个维度的大小。
        size_ = dim0;
        // 判断是否需要分配新的内存。
        if (need_alloc && alloc) {
            // 调用 allocate 方法进行内存分配。
            allocate(alloc);
        } else {
            // 如果提供了外部指针 ptr。
            if (ptr != nullptr) {
                // 确保当 ptr 不为空时，need_alloc 必须为 false。
                CHECK(need_alloc == false)
                    << "The need_alloc is true when ptr parameter is not a null pointer.";
            // 使用给定的 allocator 和数据类型初始化缓冲区。
            init_buffer(alloc, data_type_, need_alloc, ptr);
            }
        }
    }

    Tensor::Tensor(base::DataType data_type,
                   int32_t dim0,
                   int32_t dim1,
                   bool need_alloc,
                   std::shared_ptr<base::DeviceAllocator> alloc,
                   void *ptr)
        : data_type_(data_type) {
        dims_.push_back(dim0);
        dims_.push_back(dim1);
        size_ = dim0 * dim1;
        if (need_alloc && alloc) {
            allocate(alloc);
        } else {
            init_buffer(alloc, data_type_, need_alloc, ptr);
        }
    }

    Tensor::Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, bool need_alloc,
                   std::shared_ptr<base::DeviceAllocator> alloc, void *ptr)
        : data_type_(data_type) {
        dims_.push_back(dim0);
        dims_.push_back(dim1);
        dims_.push_back(dim2);
        size_ = dim0 * dim1 * dim2;
        if (need_alloc && alloc) {
            allocate(alloc);
        } else {
            init_buffer(alloc, data_type_, need_alloc, ptr);
        }
    }

    Tensor::Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, int32_t dim3,
                   bool need_alloc, std::shared_ptr<base::DeviceAllocator> alloc, void *ptr)
        : data_type_(data_type) {
        dims_.push_back(dim0);
        dims_.push_back(dim1);
        dims_.push_back(dim2);
        dims_.push_back(dim3);
        size_ = dim0 * dim1 * dim2 * dim3;
        if (need_alloc && alloc) {
            allocate(alloc);
        } else {
            init_buffer(alloc, data_type_, need_alloc, ptr);
        }
    }

    Tensor::Tensor(base::DataType data_type,
                   std::vector<int32_t> dims,
                   bool need_alloc,
                   std::shared_ptr<base::DeviceAllocator> alloc,
                   void *ptr)
        : dims_(std::move(dims)), data_type_(data_type) {
        size_ = reduce_dimension(dims_.begin(), dims_.end(), 1);
        if (need_alloc && alloc) {
            allocate(alloc);
        } else {
            init_buffer(alloc, data_type_, need_alloc, ptr);
        }
    }

    /**
     * @brief 将张量数据从当前设备迁移至 CUDA 设备。
     *
     * 该函数检查张量的当前设备类型：
     * - 如果当前设备类型未知，则记录错误日志。
     * - 如果当前设备为 CPU，则执行从 CPU 到 CUDA 的内存拷贝。
     * - 如果当前设备已经是 CUDA，则输出提示信息，不进行任何操作。
     *
     * @param stream CUDA 流，用于异步执行内存拷贝操作。
     */
    void Tensor::to_cuda(cudaStream_t stream) {
        CHECK_NE(buffer_, nullptr);
        const base::DeviceType device_type = this->device_type();
        if (device_type == base::DeviceType::kDeviceUnknown) {
            LOG(ERROR) << "The device type of the tensor is unknown.";
        } else if (device_type == base::DeviceType::kDeviceCPU) {
            size_t byte_size = this->byte_size();
            auto cu_alloc = base::CUDADeviceAllocatorFactory::get_instance();
            auto cu_buffer = std::make_shared<base::Buffer>(byte_size, cu_alloc);
            cu_alloc->memcpy(buffer_->ptr(), cu_buffer->ptr(), byte_size, base::MemcpyKind::kMemcpyCPU2CUDA,
                             stream);
            this->buffer_ = cu_buffer;
        } else {
            LOG(INFO) << "The device type of the tensor is already cuda.";
        }
    }

    void Tensor::to_cpu() {
        CHECK_NE(buffer_, nullptr);
        const base::DeviceType device_type = this->device_type();

        if (device_type == base::DeviceType::kDeviceUnknown) {
            LOG(ERROR) << "The device type of the tensor is unknown.";
        } else if (device_type == base::DeviceType::kDeviceCUDA) {
            size_t byte_size = this->byte_size();
            auto cpu_alloc = base::CPUDeviceAllocatorFactory::get_instance();
            auto cpu_buffer = std::make_shared<base::Buffer>(byte_size, cpu_alloc);
            cpu_alloc->memcpy(buffer_->ptr(), cpu_buffer->ptr(), byte_size,
                              base::MemcpyKind::kMemcpyCUDA2CPU);
            this->buffer_ = cpu_buffer;
        } else {
            LOG(INFO) << "The device type of the tensor is already cpu.";
        }
    }

    size_t Tensor::size() const { return this->size_; }

    int32_t Tensor::get_dim(int32_t idx) const {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, this->dims_.size());
        return this->dims_.at(idx);
    }

    base::DeviceType Tensor::device_type() const {
        if (!buffer_) {
            return base::DeviceType::kDeviceUnknown;
        }
        return buffer_->device_type();
    }

    bool Tensor::assign(std::shared_ptr<base::Buffer> buffer) {
        if (!buffer) {
            LOG(ERROR) << "The buffer parameter in the assign function is null pointer!";
            return false;
        }
        if (buffer_) {
            if (buffer_->device_type() != buffer->device_type()) {
                LOG(ERROR) << "The device type of the new buffer is different from the original one.";
            }
        }

        size_t byte_size = this->byte_size();
        if (byte_size > buffer->byte_size()) {
            LOG(ERROR) << "The size of buffer is too small for the tensor!";
            return false;
        }
        buffer_ = buffer;
        return true;
    }

    /**
     * @brief 分配张量所需的内存。
     *
     * @param allocator 内存分配器，用于分配设备内存。
     * @param need_realloc 是否需要强制重新分配内存。
     * @return 成功分配内存返回 true，否则返回 false。
     */
    bool Tensor::allocate(std::shared_ptr<base::DeviceAllocator> allocator,
                          bool need_realloc) {
        if (!allocator) {
            LOG(ERROR) << "The allocator parameter in the allocate function is null "
                    "pointer!";
            return false;
        }

        size_t byte_size = this->byte_size();
        if (!byte_size) {
            LOG(ERROR) << "The byte_size parameter in the allocate function is equal to zero!";
            return false;
        }

        if (buffer_ && byte_size <= buffer_->byte_size()) {
            if (!need_realloc) {
                return true;
            }
        }

        buffer_ = std::make_shared<base::Buffer>(byte_size, allocator, nullptr);
        if (!buffer_->ptr()) {
            LOG(ERROR) << "The memory allocated is a null pointer!";
            return false;
        }
        return true;
    }

    const std::vector<int32_t> &Tensor::dims() const { return this->dims_; }

    void Tensor::set_device_type(base::DeviceType device_type) const {
        if (buffer_) {
            buffer_->set_device_type(device_type);
        }
    }

    void Tensor::reset(base::DataType data_type, const std::vector<int32_t> &dims) {
        this->data_type_ = data_type;
        this->dims_ = dims;
        this->size_ = reduce_dimension(dims.begin(), dims.end(), 1);
        this->buffer_ = nullptr;
    }

    int32_t Tensor::dims_size() const { return static_cast<int32_t>(dims_.size()); }

    base::DataType Tensor::data_type() const { return data_type_; }

    void Tensor::reshape(const std::vector<int32_t> &dims) {
        size_t size = reduce_dimension(dims.begin(), dims.end(), 1);
        if (size == 0) {
            LOG(ERROR) << "Reshape to empty tensor is not allowed.";
            return;
        }
        if (!buffer_) {
            this->dims_ = dims;
            this->size_ = size;
            return;
        }

        if (size > size_) {
            auto new_buffer = std::make_shared<base::Buffer>(size * base::DataTypeSize(this->data_type_),
                                                             buffer_->allocator());
            CHECK(new_buffer->allocate());
            new_buffer->copy_from(buffer_.get());
            this->buffer_ = new_buffer;
        }
        this->dims_ = dims;
        this->size_ = size;
    }

    std::shared_ptr<base::Buffer> Tensor::get_buffer() const { return buffer_; }

    Tensor Tensor::clone() const {
        Tensor new_tensor = *this;
        size_t byte_size = this->byte_size();

        auto allocator = buffer_->allocator();
        new_tensor.buffer_ = std::make_shared<base::Buffer>(byte_size, allocator);
        new_tensor.buffer_->copy_from(buffer_.get());
        return new_tensor;
    }

    size_t Tensor::byte_size() const { return this->size() * DataTypeSize(data_type_); }


    /**
     * @brief 计算张量每个维度的步长（stride），用于在扁平化内存中定位多维索引的位置。
     *
     *        步长（stride）表示在访问张量的多维数据时，移动一个单位索引需要跨越的线性内存元素个数。
     *        例如：对于形状为 [2,3,4] 的张量，
     *            - 第0维的步长是 3*4 = 12，表示在第0维上移动一个索引位置需要跳过 12 个元素；
     *            - 第1维的步长是 4，表示在第1维上移动一个索引位置需要跳过 4 个元素；
     *            - 第2维的步长是 1，表示在第2维上移动一个索引位置即逐个访问相邻元素。
     *
     * @return 返回包含各维度步长值的向量，顺序与张量的维度顺序一致。
     */
    std::vector<size_t> Tensor::strides() const {
        std::vector<size_t> strides;
        if (!dims_.empty()) {
            for (int32_t i = 0; i < dims_.size() - 1; ++i) {
                size_t stride = reduce_dimension(dims_.begin() + i + 1, dims_.end(), 1);
                strides.push_back(stride);
            }
            strides.push_back(1);
        }
        return strides;
    }

    bool Tensor::is_empty() const {
        return size_ == 0 || buffer_ == nullptr || buffer_->ptr() == nullptr;
    }

    /**
     * @brief 初始化张量的缓冲区，根据分配器和指针选择创建新缓冲区或复用已有内存。
     *
     * @param alloc 内存分配器指针，用于分配设备内存。
     * @param data_type 数据类型，表示缓冲区中元素的类型。
     * @param need_alloc 是否需要分配新内存。
     * @param ptr 可选的外部数据指针，若提供则使用该指针初始化缓冲区。
     */
    void Tensor::init_buffer(std::shared_ptr<base::DeviceAllocator> alloc,
                             base::DataType data_type,
                             bool need_alloc, void *ptr) {
        if (!alloc && !need_alloc) {
            // 如果没有分配器且不需要分配新内存，则使用提供的指针构造一个 Buffer 对象。
            std::shared_ptr<base::Buffer> buffer =
                    std::make_shared<base::Buffer>(data_type_size(data_type) * size_, nullptr, ptr, true);
            this->buffer_ = buffer;
        } else {
            // 否则调用 allocate 方法强制重新分配内存。
            allocate(alloc, true);
        }
    }
} // namespace tensor
