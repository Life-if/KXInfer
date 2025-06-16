//
// Created by asus on 25-6-10.
//

#ifndef KXINFER_ADD_KERNEL_H
#define KXINFER_ADD_KERNEL_H

#include "tensor/tensor.h"
namespace kernel {
    void add_kernel_cpu(const tensor::Tensor& input1, const tensor::Tensor& input2,
                        const tensor::Tensor& output, void* stream = nullptr);
}  // namespace kernel

#endif //KXINFER_ADD_KERNEL_H
