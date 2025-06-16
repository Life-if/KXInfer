//
// Created by asus on 25-6-10.
//
#include <base/base.h>
#include "cpu/add_kernel.h"
#include "cuda/add_kernel.cuh"
#include "op/kernel_inference.h"

namespace kernel {
    AddKernel get_add_kernel(base::DeviceType device_type) {
        if (device_type == base::DeviceType::kDeviceCPU) {
            return add_kernel_cpu;
        } else if (device_type == base::DeviceType::kDeviceCUDA) {
            return add_kernel_cu;
        } else {
            LOG(FATAL) << "Unknown device type for get a add kernel.";
            return nullptr;
        }
    }
} // namespace kernel
