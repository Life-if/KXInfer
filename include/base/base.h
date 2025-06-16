///  Copyright (c) 2025 KongXiang
///  @file bash.h
///  @brief 基础模块定义


#ifndef KXINFER_INCLUDE_BASE_BASE_H_
#define KXINFER_INCLUDE_BASE_BASE_H_

#include <glog/logging.h>
#include <cstdint>
#include <string>

/// @brief 忽略变量或表达式的未使用警告。
///
/// 在函数参数或局部变量未被使用时，使用此宏可避免编译器警告。
/// 特别适用于回调函数、调试代码等场景。
#define UNUSED(expr) \
  do {               \
    (void)(expr);    \
  } while (0)


namespace LLMmodel {
    /**
     * @enum ModelBufferType
     * @brief 表示模型推理过程中使用的各种内存缓冲区类型。
     *
     * 该枚举用于统一标识在大语言模型（LLM）推理阶段中各个模块所使用的缓冲区，
     * 包括输入输出数据、注意力缓存、位置编码缓存等。
     *
     * 枚举值从 0 开始连续编号，便于作为数组索引使用。
     */
    enum class ModelBufferType {
        kInputTokens = 0, ///< 输入 token 缓冲区
        kInputEmbeddings = 1, ///< 输入嵌入向量缓冲区
        kOutputRMSNorm = 2, ///< 输出 RMS 归一化中间结果
        kKeyCache = 3, ///< 注意力机制中的 Key 缓存
        kValueCache = 4, ///< 注意力机制中的 Value 缓存
        kQuery = 5, ///< 查询向量缓冲区
        kInputPos = 6, ///< 输入位置索引缓冲区
        kScoreStorage = 7, ///< 注意力分数存储缓冲区
        kOutputMHA = 8, ///< 多头注意力输出缓冲区
        kAttnOutput = 9, ///< 注意力层最终输出缓冲区
        kW1Output = 10, ///< FFN 层第一个线性变换输出
        kW2Output = 11, ///< FFN 层第二个线性变换输出
        kW3Output = 12, ///< FFN 层第三个线性变换输出
        kFFNRMSNorm = 13, ///< FFN 层 RMS 归一化输出
        kForwardOutput = 15, ///< 最终输出缓冲区（GPU）
        kForwardOutputCPU = 16, ///< 最终输出缓冲区（CPU）
        kSinCache = 17, ///< Sin 位置编码缓存
        kCosCache = 18, ///< Cos 位置编码缓存
    };
} // namespace LLMmodel


namespace base {
    /**
     * @enum DeviceType
     * @brief 表示模型或张量所在的计算设备类型。
     *
     * 用于标识当前操作是在 CPU(1) 还是 GPU(2) 上进行。0为未知设备类型
     */
    enum class DeviceType : uint8_t {
        kDeviceUnknown = 0, ///< 未知设备类型
        kDeviceCPU = 1, ///< 在 CPU 上执行
        kDeviceCUDA = 2, ///< 在 CUDA 设备（GPU）上执行
    };

    /**
     * @enum DataType
     * @brief 表示数据的基本类型，通常用于张量元素的数据格式。
     */
    enum class DataType : uint8_t {
        kDataTypeUnknown = 0, ///< 未知数据类型
        kDataTypeFp32 = 1, ///< 单精度浮点数 (float)
        kDataTypeInt8 = 2, ///< 8位整型 (int8_t)
        kDataTypeInt32 = 3, ///< 32位整型 (int32_t)
    };

    /**
     * @enum ModelType
     * @brief 表示支持的模型种类。
     *
     * 当前仅支持 Llama2 模型，未来可扩展其他模型。
     */
    enum class ModelType : uint8_t {
        kModelTypeUnknown = 0, ///< 未知模型类型
        kModelTypeLLama2 = 1, ///< Llama2 系列模型
    };

    /**
     * @brief 返回指定数据类型的字节数。
     *
     * @param data_type 数据类型
     * @return size_t 数据类型的大小（以字节为单位）
     */
    inline size_t DataTypeSize(DataType data_type) {
        if (data_type == DataType::kDataTypeFp32) {
            return sizeof(float);
        } else if (data_type == DataType::kDataTypeInt8) {
            return sizeof(int8_t);
        } else if (data_type == DataType::kDataTypeInt32) {
            return sizeof(int32_t);
        } else {
            return 0;
        }
    }

    /**
     * @class NoCopyable
     * @brief 提供一个禁止拷贝构造和赋值的基类。
     *
     * 继承此类可以轻松实现非拷贝对象。
     */
    class NoCopyable {
    protected:
        NoCopyable() = default;

        ~NoCopyable() = default;

        NoCopyable(const NoCopyable &) = delete;

        NoCopyable &operator=(const NoCopyable &) = delete;
    };

    /**
     * @enum StatusCode
     * @brief 表示状态码，用于 Status 类中返回错误信息。
     */
    enum StatusCode : uint8_t {
        kSuccess = 0, ///< 成功
        kFunctionUnImplement = 1, ///< 函数未实现
        kPathNotValid = 2, ///< 路径无效
        kModelParseError = 3, ///< 模型解析失败
        kInternalError = 5, ///< 内部错误
        kKeyValueHasExist = 6, ///< 键值已存在
        kInvalidArgument = 7, ///< 参数非法
    };

    /**
     * @enum TokenizerType
     * @brief 表示分词器类型。
     */
    enum class TokenizerType {
        kEncodeUnknown = -1, ///< 未知编码方式
        kEncodeSpe = 0, ///< SPE 编码（SentencePiece）
        kEncodeBpe = 1, ///< BPE 编码（Byte Pair Encoding）
    };

    /**
     * @class Status
     * @brief 封装操作结果的状态信息，包括成功与否及错误信息。
     */
    class Status {
    public:
        /**
         * @brief 构造一个 Status 对象
         * @param code 状态码，默认为成功
         * @param err_message 错误信息，默认为空
         */
        Status(int code = StatusCode::kSuccess, std::string err_message = "");

        Status(const Status &other) = default;

        Status &operator=(const Status &other) = default;

        Status &operator=(int code);

        bool operator==(int code) const;

        bool operator!=(int code) const;

        operator int() const;

        operator bool() const;

        int32_t get_err_code() const;

        const std::string &get_err_msg() const;

        void set_err_msg(const std::string &err_msg);

    private:
        int code_ = StatusCode::kSuccess;
        std::string message_;
    };

    namespace error {

/**
 * @def STATUS_CHECK(call)
 * @brief 检查调用的 base::Status 是否为成功状态，若失败则记录日志并终止程序。
 *
 * 该宏用于简化错误检查流程，确保每次调用返回的状态对象是成功的。如果失败，则：
 * - 构建包含文件名、行号、错误码、错误信息的详细日志；
 * - 使用 glog 的 LOG(FATAL) 输出日志并终止程序。
 *
 * @note 推荐在关键路径中使用此宏进行错误处理，以保证程序健壮性。
 *
 * @param call 一个返回 base::Status 对象的表达式（通常是函数调用）
 *
 * @example
 * @code
 * STATUS_CHECK(LoadModel(model_path)); // 如果 LoadModel 返回非 success 状态，程序会终止
 * @endcode
 */
#define STATUS_CHECK(call)                                                                 \
  do {                                                                                     \
    const base::Status& status = call;                                                     \
    if (!status) {                                                                         \
      const size_t buf_size = 512;                                                         \
      char buf[buf_size];                                                                  \
      snprintf(buf, buf_size - 1,                                                          \
               "Infer error\n File:%s Line:%d\n Error code:%d\n Error msg:%s\n", __FILE__, \
               __LINE__, int(status), status.get_err_msg().c_str());                       \
      LOG(FATAL) << buf;                                                                   \
    }                                                                                      \
  } while (0)

        Status Success(const std::string &err_msg = "");

        Status FunctionNotImplement(const std::string &err_msg = "");

        Status PathNotValid(const std::string &err_msg = "");

        Status ModelParseError(const std::string &err_msg = "");

        Status InternalError(const std::string &err_msg = "");

        Status KeyHasExits(const std::string &err_msg = "");

        Status InvalidArgument(const std::string &err_msg = "");
    } // namespace error

    std::ostream &operator<<(std::ostream &os, const Status &x);
} // namespace base
#endif  // KXINFER_INCLUDE_BASE_BASE_H_
