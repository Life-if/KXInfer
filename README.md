# KXInfer
一个高效、灵活且可扩展的深度学习模型推理框架，支持主流大语言模型和视觉模型。

## 📌 项目简介

本项目设计并实现了一个高效、灵活、可扩展的深度学习模型推理框架，旨在为 LLM（如 Llama 2/3.2、Qwen）与视觉模型（如 YOLOv5、ResNet）提供快速、准确的推理服务。通过优化模型加载流程、核心算子实现及内存管理机制，显著提升了推理效率与资源利用率，能够有效支撑多种业务场景下的高性能推理需求。

## 🔧 主要特性

- ✅ 支持主流大模型（Llama 2/3.2、Qwen）和视觉模型（YOLOv5、ResNet）
- ✅ 支持 CPU 与 GPU 后端
- ✅ 提供统一接口封装，便于多模型部署
- ✅ 支持 Int8 量化模型推理，降低资源占用
- ✅ 集成 `KV Cache` 等优化机制，提升推理效率

## 🚀 技术亮点

| 特性        | 描述                                                              |
|-----------|-----------------------------------------------------------------|
| CUDA 算子优化 | 实现模型核心CUDA算子，并利用 Night-Compute / Night-System 工具进行性能分析与调优 (进行中) |
| 后端支持      | CPU / GPU 双端推理支持                                                |
| 推理加速技术    | FlashAttention、KV Cache、RMSNorm、MatMul 等 (进行中)                  |
| 量化压缩      | Int8 分组量化、AWQ 等 (进行中)                                           |

## 📈 项目成果

- 完成多个复杂 CUDA 算子开发与优化，包括 RMSNorm、MatMul、KV Cache、MultiHead Attention、FlashAttention 等；
- 推理峰值内存占用降低 [X]%，吞吐量提升 [Y]%；
- 掌握主流推理框架（llama.cpp、VLLM、TensorRT）的核心原理与优化技巧；
- 推动推理框架在资源受限环境下的部署落地，具备良好的工程实践价值与应用前景。

## 📚 参考技术

- llama.cpp
- VLLM
- TensorRT
- FlashAttention 论文：arXiv:2205.14135
- KuiperInfer

## 🧭 后续计划
- 先把这些优化点都实现，再考虑其他优化点

## 📄 开源协议