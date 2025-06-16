//
// Created by asus on 25-6-10.
//

#ifndef KXINFER_INCLUDE_OP_KERNELS_INTERFACE_H
#define KXINFER_INCLUDE_OP_KERNELS_INTERFACE_H
#include <base/cuda_config.h>
#include "tensor/tensor.h"

/**
 * @brief 定义通用操作的内核函数指针类型及获取接口。
 *
 * 提供了多种神经网络操作的底层内核函数指针定义，并封装了根据设备类型选择合适实现的接口函数。
 * 支持 CPU 和 GPU（CUDA）环境下的操作调用。
 */
namespace kernel {
    /**
     * @brief 向量加法内核函数指针。
     * @param input1 第一个输入张量
     * @param input2 第二个输入张量
     * @param output 输出张量
     * @param stream CUDA 流指针（可为 nullptr）
     */
    typedef void (*AddKernel)(const tensor::Tensor &input1, const tensor::Tensor &input2,
                              const tensor::Tensor &output, void *stream);

    /**
     * @brief 矩阵乘法内核函数指针（非量化）。
     * @param input 输入张量
     * @param weight 权重张量
     * @param output 输出张量
     * @param scale 缩放因子
     * @param config CUDA 配置参数
     */
    typedef void (*MatmulKernel)(const tensor::Tensor &input, const tensor::Tensor &weight,
                                 const tensor::Tensor &output, float scale, const CudaConfig *config);

    /**
     * @brief 量化矩阵乘法内核函数指针。
     * @param input 输入张量
     * @param weight 权重张量（已量化）
     * @param output 输出张量
     * @param group_size 量化分组大小
     * @param scale 缩放因子张量
     * @param config CUDA 配置参数
     */
    typedef void (*MatmulKernelQuant)(const tensor::Tensor &input, const tensor::Tensor &weight,
                                      const tensor::Tensor &output, int32_t group_size,
                                      const tensor::Tensor &scale, const CudaConfig *config);

    /**
     * @brief 嵌入层内核函数指针。
     * @param input 输入索引张量
     * @param weight 嵌入权重张量
     * @param output 输出张量
     * @param vocab_size 词汇表大小
     * @param stream CUDA 流指针（可为 nullptr）
     */
    typedef void (*EmbeddingKernel)(const tensor::Tensor &input, const tensor::Tensor &weight,
                                    const tensor::Tensor &output, int32_t vocab_size, void *stream);

    /**
     * @brief SwiGLU 激活函数内核函数指针。
     * @param input1 第一个输入张量（激活输入）
     * @param input2 第二个输入张量（门控输入）
     * @param output 输出张量
     * @param stream CUDA 流指针（可为 nullptr）
     */
    typedef void (*SwigluKernel)(const tensor::Tensor &input1, const tensor::Tensor &input2,
                                 const tensor::Tensor &output, void *stream);

    /**
     * @brief 多头注意力机制内核函数指针。
     * @param pos 当前位置索引
     * @param head_num 注意力头数
     * @param layer_index 层索引
     * @param seq_len 序列长度
     * @param kv_dim Key/Value 维度
     * @param kv_mul Key/Value 扩展倍数（例如是否重复头）
     * @param head_size 单头维度
     * @param mha_out 多头注意力输出张量
     * @param query_tensor Query 张量
     * @param score_tensor 注意力得分张量
     * @param key_cache_tensor Key 缓存张量
     * @param value_cache_tensor Value 缓存张量
     * @param device_type 设备类型（CPU/GPU）
     * @param config CUDA 配置参数
     */
    typedef void (*MHAKernel)(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len,
                              int32_t kv_dim, int32_t kv_mul, int32_t head_size,
                              const tensor::Tensor &mha_out, const tensor::Tensor &query_tensor,
                              const tensor::Tensor &score_tensor,
                              const tensor::Tensor &key_cache_tensor,
                              const tensor::Tensor &value_cache_tensor, base::DeviceType device_type,
                              CudaConfig *);

    /**
     * @brief RMS 归一化内核函数指针。
     * @param input 输入张量
     * @param weight 权重张量（缩放因子）
     * @param output 输出张量
     * @param stream CUDA 流指针（可为 nullptr）
     */
    typedef void (*RMSNormKernel)(const tensor::Tensor &input, const tensor::Tensor &weight,
                                  const tensor::Tensor &output, void *stream);

    /**
     * @brief RoPE（旋转位置编码）内核函数指针。
     * @param dim Query 的嵌入维度
     * @param kv_dim Key/Value 的嵌入维度
     * @param head_size 单头维度
     * @param input_q Query 张量
     * @param input_k Key 张量
     * @param input_pos 位置索引张量
     * @param sin_cache Sin 缓存张量
     * @param cos_cache Cos 缓存张量
     * @param stream CUDA 流指针（可为 nullptr）
     */
    typedef void (*RoPEKernel)(int32_t dim, int32_t kv_dim, int32_t head_size,
                               const tensor::Tensor &input_q, const tensor::Tensor &input_k,
                               const tensor::Tensor &input_pos, const tensor::Tensor &sin_cache,
                               const tensor::Tensor &cos_cache, void *stream);

    /**
     * @brief 缩放操作内核函数指针。
     * @param scale 缩放因子
     * @param input 输入张量
     * @param stream CUDA 流指针（可为 nullptr）
     */
    typedef void (*ScaleKernel)(float scale, const tensor::Tensor &input, void *stream);

    /**
      * @brief Softmax 原地计算内核函数指针。
      * @param input 输入张量（原地修改）
      * @param stream CUDA 流指针（可为 nullptr）
      */
    typedef void (*SoftmaxInplaceKernel)(const tensor::Tensor &input, void *stream);

    /**
     * @brief 缩放并求和内核函数指针。
     * @param value 要加的值
     * @param scale 缩放张量
     * @param output 输出张量
     * @param t 时间步
     * @param size 操作大小
     * @param stride 步长
     * @param stream CUDA 流指针（可为 nullptr）
     */
    typedef void (*ScaleSumKernel)(const tensor::Tensor &value, const tensor::Tensor &scale,
                                   const tensor::Tensor &output, int t, int size, int stride,
                                   void *stream);

    /**
     * @brief CPU 上执行的 Softmax 原地计算函数。
     * @param input_ptr 输入数据指针
     * @param size 数据大小
     */
    void softmax_inplace_cpu(const float *input_ptr, size_t size);

    /*
     *@brief 调用函数，返回上面的指针函数
     */

    AddKernel get_add_kernel(base::DeviceType device_type);

    EmbeddingKernel get_emb_kernel(base::DeviceType device_type);

    MatmulKernel get_matmul_kernel(base::DeviceType device_type);

    MatmulKernelQuant get_matmul_kernel_quant8(base::DeviceType device_type);

    MHAKernel get_mha_kernel(base::DeviceType device_type);

    RMSNormKernel get_rmsnorm_kernel(base::DeviceType device_type);

    RoPEKernel get_rope_kernel(base::DeviceType device_type);

    ScaleKernel get_scale_kernel(base::DeviceType device_type);

    SoftmaxInplaceKernel get_softmax_kernel(base::DeviceType device_type);

    SwigluKernel get_swiglu_kernel(base::DeviceType device_type, void *stream = nullptr);

    ScaleSumKernel get_scale_sum_kernel(base::DeviceType device_type);
} // namespace kernel
#endif  // KERNEL_INTERFACE_H
