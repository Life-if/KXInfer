//
// Created by asus on 25-6-8.
//

#include "op/layer.h"
#include <base/cuda_config.h>
#include <glog/logging.h>
#include <cstdarg>  // 用于处理可变参数列表。
#include <numeric>
#include <utility>

namespace op {
    /**
     * @brief BaseLayer 构造函数
     *
     * 初始化一个基础层对象，指定设备类型、层类型、数据类型和层名称。
     *
     * @param device_type 层运行所在的设备类型（如 CPU / GPU）
     * @param layer_type 层的类型（如 Convolution / ReLU 等）
     * @param data_type 数据精度类型（如 FP32 / INT8）
     * @param layer_name 层的名称
     */
    BaseLayer::BaseLayer(base::DeviceType device_type, LayerType layer_type, base::DataType data_type,
                         std::string layer_name)
        : device_type_(device_type),
          layer_type_(layer_type),
          data_type_(data_type),
          layer_name_(std::move(layer_name)) {
    }

    base::DataType BaseLayer::data_type() const { return data_type_; }

    LayerType BaseLayer::layer_type() const { return layer_type_; }

    /**
     * @brief 设置权重张量（未实现）
     *
     * 子类应重写此方法以实现具体设置逻辑。
     *
     * @param idx 权重索引
     * @param weight 要设置的张量
     * @return 错误状态
     */
    base::Status BaseLayer::set_weight(int32_t idx, const tensor::Tensor &weight) {
        return base::error::FunctionNotImplement();
    }

    /**
     * @brief 使用内存指针设置权重（未实现）
     *
     * 子类应重写此方法以实现具体设置逻辑。
     *
     * @param idx 权重索引
     * @param dims 张量维度
     * @param weight_ptr 数据指针
     * @param device_type 设备类型
     * @return 错误状态
     */
    base::Status BaseLayer::set_weight(int32_t idx, const std::vector<int32_t> &dims,
                                       const void *weight_ptr, base::DeviceType device_type) {
        return base::error::FunctionNotImplement();
    }

    /**
     * @brief 获取当前层的名称
     *
     * @return 层名称
     */
    const std::string &BaseLayer::get_layer_name() const { return layer_name_; }


    void BaseLayer::set_layer_name(const std::string &layer_name) { layer_name_ = layer_name; }

    base::DeviceType BaseLayer::device_type() const { return device_type_; }

    void BaseLayer::set_device_type(base::DeviceType device_type) { device_type_ = device_type; }

    // ---------------------------------------------------------------------------

    /**
     * @brief Layer 构造函数
     *
     * 默认使用 FP32 精度构造一个层。
     *
     * @param device_type 设备类型
     * @param layer_type 层类型
     * @param layer_name 层名称
     */
    Layer::Layer(base::DeviceType device_type, LayerType layer_type, std::string layer_name)
        : BaseLayer(device_type, layer_type, base::DataType::kDataTypeFp32, std::move(layer_name)) {
    }

    base::Status Layer::init() { return base::error::Success(); }

    /**
     * @brief 前向传播（未实现）
     *
     * 子类需实现具体的前向计算逻辑。
     *
     * @return 错误状态
     */
    base::Status Layer::forward() { return base::error::FunctionNotImplement(""); }


    base::Status Layer::check_tensor(const tensor::Tensor &tensor, base::DeviceType device_type,
                                     base::DataType data_type) const {
        if (tensor.is_empty()) {
            return base::error::InvalidArgument("The tensor parameter is empty.");
        }
        if (tensor.device_type() != device_type) {
            return base::error::InvalidArgument("The tensor has a wrong device type.");
        }
        if (tensor.data_type() != data_type) {
            return base::error::InvalidArgument("The tensor has a wrong data type.");
        }
        return base::error::Success();
    }

    /**
     * @brief 校验张量的维度信息
     *
     * 支持可变参数校验每个维度是否符合预期。
     *
     * @param tensor 待检查的张量
     * @param device_type 预期设备类型
     * @param data_type 预期数据类型
     * @param check_dims   预期的维度值列表
     * @return 校验结果
     */
    base::Status Layer::check_tensor_with_dim(const tensor::Tensor &tensor,
                                              base::DeviceType device_type,
                                              base::DataType data_type,
                                              std::vector<int32_t> &check_dims) const {
        // std::va_list args;
        if (tensor.is_empty()) {
            return base::error::InvalidArgument("The tensor parameter is empty.");
        }
        if (tensor.device_type() != device_type) {
            return base::error::InvalidArgument("The tensor has a wrong device type.");
        }
        if (tensor.data_type() != data_type) {
            return base::error::InvalidArgument("The tensor has a wrong data type.");
        }

        int32_t dims = tensor.dims_size();
        if (dims != check_dims.size()) {
            return base::error::InvalidArgument("The tensor has a wrong dim size. Except:" + std::to_string(dims) +
                                                "But get " + std::to_string(check_dims.size()));
        }
        for (auto &check_d: check_dims) {
            if (check_d < 0) {
                return base::error::InvalidArgument("The tensor has a wrong dim :" + std::to_string(check_d));
            }
        }

        for (int32_t i = 0; i < dims; ++i) {
            if (check_dims.at(i) != tensor.get_dim(i)) {
                return base::error::InvalidArgument("The tensor has a wrong dim in dim" + std::to_string(i));
            }
        }

        return base::error::Success();
    }


    void Layer::set_input(int32_t idx, const tensor::Tensor &input) {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, inputs_.size());
        this->inputs_.at(idx) = input;
    }

    void Layer::set_output(int32_t idx, const tensor::Tensor &output) {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, outputs_.size());
        this->outputs_.at(idx) = output;
    }

    const tensor::Tensor &Layer::get_input(int32_t idx) const {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, inputs_.size());
        return inputs_.at(idx);
    }

    tensor::Tensor &Layer::get_input(int32_t idx) {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, inputs_.size());
        return inputs_.at(idx);
    }

    tensor::Tensor &Layer::get_output(int32_t idx) {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, outputs_.size());
        return outputs_.at(idx);
    }

    base::Status Layer::check() const {
        return base::error::FunctionNotImplement("The check function is not implement yet");
    }

    const tensor::Tensor &Layer::get_output(int32_t idx) const {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, outputs_.size());
        return outputs_.at(idx);
    }

    void Layer::reset_input_size(size_t size) { inputs_.resize(size); }

    void Layer::reset_output_size(size_t size) { outputs_.resize(size); }

    void Layer::to_cuda() {
        for (auto &input: inputs_) {
            if (!input.is_empty()) {
                input.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
            }
        }
        for (auto &output: outputs_) {
            if (!output.is_empty()) {
                output.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
            }
        }
    }

    void Layer::set_cuda_config(std::shared_ptr<kernel::CudaConfig> config) {
        if (!config) {
            return;
        }
        this->cuda_config_ = config;
    }

    std::shared_ptr<kernel::CudaConfig> Layer::cuda_config() const { return cuda_config_; }

    size_t Layer::input_size() const { return inputs_.size(); }

    size_t Layer::output_size() const { return outputs_.size(); }

    //------------------------------------------------------------------------------------------

    LayerParam::LayerParam(base::DeviceType device_type,
                           LayerType layer_type,
                           bool is_quant_layer,
                           std::string layer_name)
        : Layer(device_type, layer_type, std::move(layer_name)), is_quant_layer_(is_quant_layer) {
    }

    base::Status LayerParam::set_weight(int32_t idx, const tensor::Tensor &weight) {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, weights_.size());
        CHECK(weight.data_type() == base::DataType::kDataTypeFp32);
        if (!weight.is_empty()) {
            CHECK(weight.device_type() == device_type_);
        }
        weights_.at(idx) = weight;
        return base::error::Success();
    }

    const tensor::Tensor &LayerParam::get_weight(int32_t idx) const {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, weights_.size());
        return weights_.at(idx);
    }

    void LayerParam::to_cuda() {
        Layer::to_cuda();
        for (auto &weight: weights_) {
            weight.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
        }
        if (!scales_.is_empty()) {
            scales_.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
        }
    }

    /**
     * @brief 根据维度和指针设置指定索引处的权重张量
     * @param idx 权重索引
     * @param dims 张量维度
     * @param weight_ptr 权重数据指针
     * @param device_type 数据所在的设备类型，默认为未知
     * @return 操作状态码
     */
    base::Status LayerParam::set_weight(int32_t idx,
                                        const std::vector<int32_t> &dims,
                                        const void *weight_ptr,
                                        base::DeviceType device_type) {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, weights_.size());
        CHECK_NE(weight_ptr, nullptr);

        size_t size = std::accumulate(dims.begin(), dims.end(), sizeof(float), std::multiplies<>());
        std::shared_ptr<base::Buffer> buffer =
                std::make_shared<base::Buffer>(size, nullptr, const_cast<void *>(weight_ptr), true);

        if (device_type != base::DeviceType::kDeviceUnknown) {
            buffer->set_device_type(device_type);
        }

        if (!is_quant_layer_) {
            tensor::Tensor weight(base::DataType::kDataTypeFp32, dims);
            weight.set_device_type(device_type);
            CHECK(weight.assign(buffer));
            weights_.at(idx) = weight;
        } else {
            // is quant layer
            tensor::Tensor weight(base::DataType::kDataTypeInt8, dims);
            weight.set_device_type(device_type);
            CHECK(weight.assign(buffer));
            weights_.at(idx) = weight;


            // 获取 INT8 权重的总元素数量；
            // 检查是否可以整除 group_size_（每个 group 共享一个 scale）；
            const int32_t weight_size = static_cast<int32_t>(weight.size());
            CHECK(weight_size % group_size_ == 0);

            int32_t scale_nums = weight_size / group_size_;


            scales_ = tensor::Tensor{
                base::DataType::kDataTypeFp32, scale_nums, false, nullptr,
                reinterpret_cast<float *>((int8_t *) weight_ptr + weight_size)
            };
            scales_.set_device_type(device_type);
        }

        return base::error::Success();
    }

    void LayerParam::set_scales(const tensor::Tensor &scales) {
        CHECK(!scales.is_empty());
        this->scales_ = scales;
    }

    void LayerParam::set_group_size(int32_t group_size) { this->group_size_ = group_size; }

    int32_t LayerParam::get_scale_num() const {
        CHECK(!scales_.is_empty());
        return static_cast<int32_t>(scales_.size());
    }

    void LayerParam::reset_weight_size(size_t size) { weights_.resize(size); }

    size_t LayerParam::weight_size() const { return weights_.size(); }

    base::Status Layer::forward(const tensor::Tensor &input1, const tensor::Tensor &output1) {
        this->set_input(0, input1);
        this->set_output(0, output1);
        return this->forward();
    }

    base::Status Layer::forward(const tensor::Tensor &input1, const tensor::Tensor &input2,
                                const tensor::Tensor &output1) {
        this->set_input(0, input1);
        this->set_input(1, input2);

        this->set_output(0, output1);
        return this->forward();
    }

    base::Status Layer::forward(const tensor::Tensor &input1, const tensor::Tensor &input2,
                                const tensor::Tensor &input3, const tensor::Tensor &output1) {
        this->set_input(0, input1);
        this->set_input(1, input2);
        this->set_input(2, input3);

        this->set_output(0, output1);
        return this->forward();
    }

    base::Status Layer::forward(const tensor::Tensor &input1, const tensor::Tensor &input2,
                                const tensor::Tensor &input3, const tensor::Tensor &input4,
                                const tensor::Tensor &output1) {
        this->set_input(0, input1);
        this->set_input(1, input2);
        this->set_input(2, input3);
        this->set_input(3, input4);

        this->set_output(0, output1);
        return this->forward();
    }

    base::Status Layer::forward(const tensor::Tensor &input1, const tensor::Tensor &input2,
                                const tensor::Tensor &input3, const tensor::Tensor &input4,
                                const tensor::Tensor &input5, const tensor::Tensor &output1) {
        this->set_input(0, input1);
        this->set_input(1, input2);
        this->set_input(2, input3);
        this->set_input(3, input4);
        this->set_input(4, input5);

        this->set_output(0, output1);
        return this->forward();
    }

    tensor::Tensor &LayerParam::get_weight(int32_t idx) {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, weights_.size());
        return weights_.at(idx);
    }
} // namespace op
