//
// Created by asus on 25-6-9.
//

#ifndef KXINFER_INCLUDE_BASE_CUDA_CONFIG_H
#define KXINFER_INCLUDE_BASE_CUDA_CONFIG_H
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

namespace kernel {
    /**
     * @brief CUDA配置管理类
     *
     * 该结构体用于管理CUDA流（stream）的生命周期，
     * 在析构时自动销毁关联的CUDA流资源。
     */
    struct CudaConfig {
        /**
         * @brief 默认构造CUDA流对象
         */
        cudaStream_t stream = nullptr;

        /**
         * @brief 析构函数
         *
         * 如果当前对象持有有效的CUDA流，则在对象销毁时
         * 自动调用cudaStreamDestroy释放资源，避免内存泄漏。
         */
        ~CudaConfig() {
            if (stream) {
                cudaStreamDestroy(stream);
            }
        }
    };
}  // namespace kernel
#endif //KXINFER_INCLUDE_BASE_CUDA_CONFIG_H
