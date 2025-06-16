//
// Created by asus on 25-6-8.
//

#ifndef KXINFER_INCLUDE_TENSOR_TENSOR_H_
#define KXINFER_INCLUDE_TENSOR_TENSOR_H_
#include <driver_types.h>   // CUDA 类型定义
#include <glog/logging.h>
#include <armadillo>
#include <memory>
#include <vector>
#include "base/base.h"
#include "base/buffer.h"

namespace tensor {
    class Tensor {
    public:
        explicit Tensor() = default;

        /**
         * @brief 构造一个一维张量
         * @param data_type 数据类型
         * @param dim0 第一维度大小
         * @param need_alloc 是否需要分配内存
         * @param alloc 内存分配器（可为空）
         * @param ptr 已有数据指针（可为空）
         */
        explicit Tensor(base::DataType data_type,
                        int32_t dim0,
                        bool need_alloc = false,
                        std::shared_ptr<base::DeviceAllocator> alloc = nullptr,
                        void *ptr = nullptr);

        explicit Tensor(base::DataType data_type,
                        int32_t dim0,
                        int32_t dim1,
                        bool need_alloc = false,
                        std::shared_ptr<base::DeviceAllocator> alloc = nullptr,
                        void *ptr = nullptr);

        explicit Tensor(base::DataType data_type,
                        int32_t dim0,
                        int32_t dim1,
                        int32_t dim2,
                        bool need_alloc = false, std::shared_ptr<base::DeviceAllocator> alloc = nullptr,
                        void *ptr = nullptr);

        explicit Tensor(base::DataType data_type,
                        int32_t dim0,
                        int32_t dim1,
                        int32_t dim2,
                        int32_t dim3,
                        bool need_alloc = false, std::shared_ptr<base::DeviceAllocator> alloc = nullptr,
                        void *ptr = nullptr);

        /**
         * @brief 使用维度向量构造任意维度的张量
         * @param data_type 数据类型
         * @param dims 维度向量
         * @param need_alloc 是否需要分配内存
         * @param alloc 分配器
         * @param ptr 指针（可为空）
         */
        explicit Tensor(base::DataType data_type,
                        std::vector<int32_t> dims,
                        bool need_alloc = false,
                        std::shared_ptr<base::DeviceAllocator> alloc = nullptr,
                        void *ptr = nullptr);

        void to_cpu();

        void to_cuda(cudaStream_t stream = nullptr);

        bool is_empty() const;

        /**
         * @brief 初始化内部缓冲区
         * @param alloc 分配器
         * @param data_type 数据类型
         * @param need_alloc 是否需要分配内存
         * @param ptr 外部数据指针（可为空）
         */
        void init_buffer(std::shared_ptr<base::DeviceAllocator> alloc,
                         base::DataType data_type,
                         bool need_alloc,
                         void *ptr);

        /**
         * @brief 获取指向数据的模板化指针
         * @tparam T 数据类型
         * @return 数据指针
         */
        template<typename T>
        T *ptr();

        template<typename T>
        const T *ptr() const;

        void reshape(const std::vector<int32_t> &dims);

        /**
         * @brief 获取底层缓冲区
         * @return 缓冲区智能指针
         */
        std::shared_ptr<base::Buffer> get_buffer() const;

        /**
         * @brief 获取元素总数
         * @return 元素数量
         */
        size_t size() const;

        /**
         * @brief 获取字节大小
         * @return 字节数
         */
        size_t byte_size() const;

        int32_t dims_size() const;

        base::DataType data_type() const;

        /**
         * @brief 获取指定索引的维度值
         * @param idx 维度索引
         * @return 维度值
         */
        int32_t get_dim(int32_t idx) const;

        const std::vector<int32_t> &dims() const;

        /**
         * @brief 计算各维度步长
         * @return 步长数组
         */
        std::vector<size_t> strides() const;

        /**
         * @brief 替换底层缓冲区
         * @param buffer 新缓冲区
         * @return 是否成功
         */
        bool assign(std::shared_ptr<base::Buffer> buffer);

        /**
         * @brief 重置张量为新类型和维度
         * @param data_type 新数据类型
         * @param dims 新维度
         */
        void reset(base::DataType data_type, const std::vector<int32_t> &dims);

        /**
         * @brief 设置设备类型
         * @param device_type 新设备类型
         */
        void set_device_type(base::DeviceType device_type) const;


        base::DeviceType device_type() const;

        /**
         * @brief 分配内存
         * @param allocator 分配器
         * @param need_realloc 是否强制重新分配
         * @return 是否成功
         */
        bool allocate(std::shared_ptr<base::DeviceAllocator> allocator,
                      bool need_realloc = false);

        /**
         * @brief 获取指定索引位置的数据指针（带类型转换）
         *
         * 用于访问张量内部缓冲区的特定元素，通过偏移索引定位。
         *
         * @tparam T 数据类型（如 float, int 等）
         * @param index 元素索引（从0开始）
         * @return 指向该位置的模板化指针
         *
         * @note 如果 buffer 为空或数据指针为 nullptr，则触发 CHECK 断言失败
         */
        template<typename T>
        T *ptr(int64_t index);

        template<typename T>
        const T *ptr(int64_t index) const;


        /**
         * @brief 获取指定偏移位置的引用（带类型转换和边界检查）
         *
         * 使用偏移量直接访问数据缓冲区中的元素。
         *
         * @tparam T 数据类型
         * @param offset 数据缓冲区内的字节偏移量
         * @return 对应位置的引用
         *
         * @throws CHECK(offset >= 0 && offset < size()) 若越界则断言失败
         */
        template<typename T>
        T &index(int64_t offset);

        template<typename T>
        const T &index(int64_t offset) const;

        /**
         * @brief 创建当前张量的深拷贝（克隆）
         *
         * 返回一个与当前张量具有相同维度、数据类型和内容的新张量实例。
         *
         * @return 新的 Tensor 实例
         */
        tensor::Tensor clone() const;

    private:
        size_t size_ = 0;                      ///< 总元素数
        std::vector<int32_t> dims_;           ///< 各维度大小
        std::shared_ptr<base::Buffer> buffer_;///< 数据缓冲区
        base::DataType data_type_;            ///< 数据类型
    };

    template<typename T>
    T &Tensor::index(int64_t offset) {
        CHECK_GE(offset, 0);
        CHECK_LT(offset, this->size());   // 检查 offset < 张量总元素数
        T &val = *(reinterpret_cast<T *>(buffer_->ptr()) + offset);
        return val;
    }

    template<typename T>
    const T &Tensor::index(int64_t offset) const {
        CHECK_GE(offset, 0);
        CHECK_LT(offset, this->size());
        const T &val = *(reinterpret_cast<T *>(buffer_->ptr()) + offset);
        return val;
    }

    template<typename T>
    const T *Tensor::ptr() const {
        if (!buffer_) {
            return nullptr;
        }
        return const_cast<const T *>(reinterpret_cast<T *>(buffer_->ptr()));
    }

    template<typename T>
    T *Tensor::ptr() {
        if (!buffer_) {
            return nullptr;
        }
        return reinterpret_cast<T *>(buffer_->ptr());
    }

    template<typename T>
    T *Tensor::ptr(int64_t index) {
        CHECK(buffer_ != nullptr && buffer_->ptr() != nullptr)
      << "The data area buffer of this tensor is empty or it points to a null pointer.";
        return const_cast<T *>(reinterpret_cast<const T *>(buffer_->ptr())) + index;
    }

    template<typename T>
    const T *Tensor::ptr(int64_t index) const {
        CHECK(buffer_ != nullptr && buffer_->ptr() != nullptr)
      << "The data area buffer of this tensor is empty or it points to a null pointer.";
        return reinterpret_cast<const T *>(buffer_->ptr()) + index;
    }
} // namespace tensor
#endif  // KXINFER_INCLUDE_TENSOR_TENSOR_H_
