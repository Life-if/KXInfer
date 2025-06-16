//
// Created by asus on 25-6-8.
//

#ifndef KXINFER_INCLUDE_OP_LAYER_H_
#define KXINFER_INCLUDE_OP_LAYER_H_
#include <base/cuda_config.h>
#include <string>
#include <vector>
#include "base/base.h"
#include "tensor/tensor.h"

namespace op {
    /**
     * @brief 表示不同类型的神经网络操作层。
     */
    enum class LayerType : uint8_t {
        kLayerUnknown = 0, ///< 未知层类型
        kLayerLinear = 1, ///< 线性变换层（全连接）
        kLayerEncode = 2, ///< 编码层
        kLayerEmbedding = 3, ///< 嵌入层
        kLayerRMSNorm = 4, ///< RMS 归一化层
        kLayerMatmul = 5, ///< 矩阵乘法层
        kLayerRoPe = 6, ///< RoPE（旋转位置编码）
        kLayerMHA = 7, ///< 多头注意力机制
        kLayerSoftmax = 8, ///< Softmax 层
        kLayerAdd = 9, ///< 加法层
        kLayerSwiGLU = 10, ///< SwiGLU 激活函数层
    };

    /**
     * @brief 所有层的基类，提供统一接口和基本属性。
     *
     * 包含数据类型、设备类型、层名称等通用信息。
     */
    class BaseLayer {
    public:
        /**
         * @brief 构造函数
         * @param device_type 层运行的设备类型（CPU/GPU）
         * @param layer_type 层的类型
         * @param data_type 数据类型（如 float, int 等）
         * @param layer_name 层的名称（可选）
         */
        explicit BaseLayer(base::DeviceType device_type,
                           LayerType layer_type,
                           base::DataType data_type,
                           std::string layer_name = "");

        /**
         * @brief 获取当前层的数据类型
         * @return 当前数据类型
         */
        base::DataType data_type() const;

        /**
         * @brief 获取当前层的类型
         * @return 层类型
         */
        LayerType layer_type() const;

        virtual base::Status init() = 0;

        virtual base::Status forward() = 0;

        virtual base::Status forward(const tensor::Tensor &input1, const tensor::Tensor &output1) = 0;

        virtual base::Status forward(const tensor::Tensor &input1, const tensor::Tensor &input2,
                                     const tensor::Tensor &output1) = 0;

        virtual base::Status forward(const tensor::Tensor &input1, const tensor::Tensor &input2,
                                     const tensor::Tensor &input3, const tensor::Tensor &output1) = 0;

        virtual base::Status forward(const tensor::Tensor &input1, const tensor::Tensor &input2,
                                     const tensor::Tensor &input3, const tensor::Tensor &input4,
                                     const tensor::Tensor &output1) = 0;

        virtual base::Status forward(const tensor::Tensor &input1, const tensor::Tensor &input2,
                                     const tensor::Tensor &input3, const tensor::Tensor &input4,
                                     const tensor::Tensor &input5, const tensor::Tensor &output1) = 0;

        virtual void set_input(int32_t idx, const tensor::Tensor &input) = 0;

        virtual void set_output(int32_t idx, const tensor::Tensor &output) = 0;

        virtual size_t input_size() const = 0;

        virtual size_t output_size() const = 0;

        virtual base::Status check() const = 0;

        virtual tensor::Tensor &get_input(int32_t idx) = 0;

        virtual tensor::Tensor &get_output(int32_t idx) = 0;

        virtual const tensor::Tensor &get_input(int32_t idx) const = 0;

        virtual const tensor::Tensor &get_output(int32_t idx) const = 0;

        virtual base::Status set_weight(int32_t idx, const tensor::Tensor &weight);

        virtual base::Status set_weight(int32_t idx, const std::vector<int32_t> &dims,
                                        const void *weight_ptr,
                                        base::DeviceType device_type = base::DeviceType::kDeviceUnknown);

        const std::string &get_layer_name() const;

        void set_layer_name(const std::string &layer_name);

        base::DeviceType device_type() const;

        void set_device_type(base::DeviceType device_type);

    protected:
        std::string layer_name_;
        LayerType layer_type_ = LayerType::kLayerUnknown;
        base::DataType data_type_ = base::DataType::kDataTypeUnknown;
        base::DeviceType device_type_ = base::DeviceType::kDeviceUnknown;
    };


    class Layer : public BaseLayer {
    public:
        explicit Layer(base::DeviceType device_type, LayerType layer_type, std::string layer_name = "");

        base::Status init() override;

        base::Status check_tensor(const tensor::Tensor &tensor, base::DeviceType device_type,
                                  base::DataType data_type) const;

        base::Status check_tensor_with_dim(const tensor::Tensor &tensor, base::DeviceType device_type,
                                           base::DataType data_type, std::vector<int32_t>&check_dims ) const;

        base::Status check() const override;

        base::Status forward() override;

        base::Status forward(const tensor::Tensor &input1, const tensor::Tensor &output1) override;

        base::Status forward(const tensor::Tensor &input1, const tensor::Tensor &input2,
                             const tensor::Tensor &output1) override;

        base::Status forward(const tensor::Tensor &input1, const tensor::Tensor &input2,
                             const tensor::Tensor &input3, const tensor::Tensor &output1) override;

        base::Status forward(const tensor::Tensor &input1, const tensor::Tensor &input2,
                             const tensor::Tensor &input3, const tensor::Tensor &input4,
                             const tensor::Tensor &output1) override;

        base::Status forward(const tensor::Tensor &input1, const tensor::Tensor &input2,
                             const tensor::Tensor &input3, const tensor::Tensor &input4,
                             const tensor::Tensor &input5, const tensor::Tensor &output1) override;

        void set_input(int32_t idx, const tensor::Tensor &input) override;

        void set_output(int32_t idx, const tensor::Tensor &output) override;

        const tensor::Tensor &get_input(int32_t idx) const override;

        const tensor::Tensor &get_output(int32_t idx) const override;

        tensor::Tensor &get_input(int32_t idx) override;

        tensor::Tensor &get_output(int32_t idx) override;

        size_t input_size() const override;

        size_t output_size() const override;

        void reset_input_size(size_t size);

        void reset_output_size(size_t size);

        virtual void to_cuda();

        void set_cuda_config(std::shared_ptr<kernel::CudaConfig> config);

        std::shared_ptr<kernel::CudaConfig> cuda_config() const;

    protected:
        std::vector<tensor::Tensor> inputs_;
        std::vector<tensor::Tensor> outputs_;
        std::shared_ptr<kernel::CudaConfig> cuda_config_;
    };

    /**
     * @brief 参数化层类，继承自Layer，用于管理带有权重参数的神经网络层。
     *
     * LayerParam 提供了对模型中可学习参数（如权重、缩放因子等）的统一管理接口，
     * 支持设置和获取权重张量、配置量化相关参数等功能。
     */
    class LayerParam : public Layer {
    public:
        /**
         * @brief 构造函数
         * @param device_type 层运行的设备类型（CPU/GPU）
         * @param layer_type 当前层的具体类型（如线性层、卷积层等）
         * @param is_quant_layer 是否为量化层，默认为false
         * @param layer_name 层的名称（可选）
         */
        explicit LayerParam(base::DeviceType device_type, LayerType layer_type,
                            bool is_quant_layer = false, std::string layer_name = "");

        /**
         * @brief 获取当前层的权重数量
         * @return 权重张量的数量
         */
        size_t weight_size() const;

        /**
         * @brief 重置权重向量的大小
         * @param size 新的权重数量
         */
        void reset_weight_size(size_t size);

        /**
         * @brief 获取指定索引处的权重张量（可修改）
         * @param idx 权重索引
         * @return 指定索引处的权重张量
         */
        tensor::Tensor &get_weight(int32_t idx);

        const tensor::Tensor &get_weight(int32_t idx) const;

        /**
         * @brief 将当前层的所有数据迁移到GPU上
         */
        void to_cuda() override;

        /**
         * @brief 设置指定索引处的权重张量
         * @param idx 权重索引
         * @param weight 要设置的权重张量
         * @return 操作状态码
         */
        base::Status set_weight(int32_t idx, const tensor::Tensor &weight) override;

        /**
         * @brief 根据维度和指针设置指定索引处的权重张量
         * @param idx 权重索引
         * @param dims 张量维度
         * @param weight_ptr 权重数据指针
         * @param device_type 数据所在的设备类型，默认为未知
         * @return 操作状态码
         */
        base::Status set_weight(int32_t idx, const std::vector<int32_t> &dims, const void *weight_ptr,
                                base::DeviceType device_type = base::DeviceType::kDeviceUnknown) override;

        /**
         * @brief 设置缩放因子张量
         * @param scales 缩放因子张量
         */
        void set_scales(const tensor::Tensor &scales);

        /**
         * @brief 设置分组大小
         * @param group_size 分组大小值
         */
        void set_group_size(int32_t group_size);

        /**
         * @brief 获取缩放因子的数量
         * @return 缩放因子的数量
         */
        int32_t get_scale_num() const;

    protected:
        int32_t group_size_ = 0; ///< 量化时使用的分组大小
        bool is_quant_layer_ = false; ///< 标识是否为量化层
        tensor::Tensor scales_; ///< 缩放因子张量
        std::vector<tensor::Tensor> weights_; ///< 权重张量列表
    };
} // namespace op
#endif  // KXINFER_INCLUDE_OP_LAYER_H_
