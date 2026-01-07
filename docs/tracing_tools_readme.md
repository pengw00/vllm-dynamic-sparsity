# vLLM 推理路径追踪工具

这些脚本帮助你追踪 vLLM 推理过程中的算子调用和 CUDA kernel 执行。

## 📁 文件说明

### 1. `test_simple_trace.py` - 简单追踪（推荐）

**功能**：
- 追踪每个 Layer 的调用
- 统计模块调用次数
- 显示执行路径

**使用**：
```bash
python test_simple_trace.py
```

**输出示例**：
```
📋 模块调用次数:
  • RMSNorm: 48 次
  • Attention: 24 次
  • MLP: 24 次
  • QKVParallelLinear: 24 次
  ...

第一层 Transformer 的调用顺序:
 1. [RMSNorm] input_layernorm
     Input:  (1, 7, 1536)
     Output: (1, 7, 1536)
 2. [QKVParallelLinear] qkv_proj
     Input:  (1, 7, 1536)
     Output: (1, 7, 4608)
 ...
```

---

### 2. `test_inference_with_logs.py` - 详细追踪

**功能**：
- 更详细的模块信息
- 调用栈追踪
- 按模块类型分组统计

**使用**：
```bash
python test_inference_with_logs.py
```

**输出示例**：
```
🔸 [RMSNorm] model.layers.0.input_layernorm
   ├─ Input: (1, 7, 1536)
   └─ Output: (1, 7, 1536)

🔸 [Qwen2Attention] model.layers.0.self_attn
   ├─ Input: (1, 7, 1536)
   └─ Output: (1, 7, 1536)
   
Top 10 最频繁调用的模块:
  1. RMSNorm:model.layers.0.input_layernorm: 2 次
  2. Attention:model.layers.0.self_attn: 2 次
  ...
```

---

### 3. `test_cuda_profiler.py` - CUDA Kernel 追踪（最详细）

**功能**：
- 使用 PyTorch Profiler
- 追踪每个 CUDA kernel 的调用
- 显示 kernel 耗时和调用次数
- 识别 vLLM 核心 kernels

**使用**：
```bash
python test_cuda_profiler.py
```

**输出示例**：
```
🔥 CUDA Kernel 调用统计（Top 30）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 1. paged_attention_v2_kernel
     Total:    15.234 ms | Calls:   24 | Avg:  0.635 ms
     
 2. fused_add_rms_norm_kernel
     Total:     8.456 ms | Calls:   48 | Avg:  0.176 ms
     
 3. rotary_embedding_kernel
     Total:     3.123 ms | Calls:   24 | Avg:  0.130 ms
     
 4. silu_and_mul_kernel
     Total:     2.789 ms | Calls:   24 | Avg:  0.116 ms
     
 ...

🎯 vLLM 核心 Kernels
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. PagedAttention Kernels:
   • paged_attention_v2_kernel
     Calls: 24, Total: 15.234 ms
   • paged_attention_v2_reduce_kernel
     Calls: 24, Total: 2.123 ms

2. RMSNorm Kernels:
   • fused_add_rms_norm_kernel
     Calls: 48, Total: 8.456 ms

3. Rotary Embedding Kernels:
   • rotary_embedding_kernel
     Calls: 24, Total: 3.123 ms

4. SiLU Activation Kernels:
   • silu_and_mul_kernel
     Calls: 24, Total: 2.789 ms
```

**额外输出**：
- 生成 `profiler_report.txt` - 完整的 profiler 报告（100+ kernels）

---

## 🎯 使用场景

### 场景 1：快速了解执行路径
```bash
python test_simple_trace.py
```
→ 看到每个 Layer 的调用顺序

### 场景 2：调试特定模块
```bash
python test_inference_with_logs.py
```
→ 详细的输入输出形状，方便调试

### 场景 3：性能分析
```bash
python test_cuda_profiler.py
```
→ 找出性能瓶颈的 CUDA kernel

---

## 📊 理解输出

### 每个 Token 生成的算子调用顺序

```
对于 24 层的模型，每生成 1 个 token：

循环 24 次（每层）：
  1. RMSNorm (Pre-Attention)     → CUDA: rms_norm_kernel
  2. QKV Projection               → cuBLAS: gemm
  3. Rotary Embedding             → CUDA: rotary_embedding_kernel
  4. PagedAttention               → CUDA: paged_attention_v2_kernel
                                         paged_attention_v2_reduce_kernel
  5. Output Projection            → cuBLAS: gemm
  6. RMSNorm (Post-Attention)     → CUDA: rms_norm_kernel
  7. Gate+Up Projection (MLP)     → cuBLAS: gemm
  8. SiLU Activation              → CUDA: silu_and_mul_kernel
  9. Down Projection (MLP)        → cuBLAS: gemm

总计每个 token：
- RMSNorm: 48 次（24 层 × 2）
- PagedAttention: 24 次（24 层 × 1）
- Rotary Embedding: 24 次
- SiLU: 24 次
- Linear (cuBLAS): 96 次（24 层 × 4）
```

### CUDA Kernel 到源码的映射

| Kernel 名称 | 源码位置 |
|------------|---------|
| `paged_attention_v2_kernel` | `csrc/attention/paged_attention_v2.cu` |
| `paged_attention_v2_reduce_kernel` | `csrc/attention/paged_attention_v2.cu` |
| `fused_add_rms_norm_kernel` | `csrc/ops/layernorm.cu` |
| `rotary_embedding_kernel` | `csrc/ops/rotary_embedding.cu` |
| `silu_and_mul_kernel` | `csrc/ops/activation.cu` |

---

## 🔧 自定义追踪

### 追踪特定模块

修改 `test_simple_trace.py`：

```python
# 在 track_layer_calls 函数中修改过滤条件

# 只追踪 Attention
if 'Attention' in module_type:
    hook = module.register_forward_hook(...)

# 只追踪 MLP
if 'MLP' in module_type:
    hook = module.register_forward_hook(...)

# 追踪你的自定义模块
if 'SonicMoE' in module_type:
    hook = module.register_forward_hook(...)
```

### 修改生成参数

```python
# 生成更多 tokens
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=100,  # 改这里
)

# 使用不同的 prompt
prompts = ["Your custom prompt here"]
```

---

## 💡 常见问题

### Q1: 为什么看不到具体的 CUDA kernel 名称？

A: 使用 `test_cuda_profiler.py`，它会显示所有 CUDA kernel。

### Q2: 如何追踪我添加的新算子（如 SonicMoE）？

A: 修改追踪脚本中的 `key_modules` 列表：

```python
key_modules = [
    'RMSNorm', 'Attention', 'MLP', 
    'SonicMoE',  # ← 添加你的模块
]
```

### Q3: 输出太多，如何过滤？

A: 修改脚本中的显示数量：

```python
# 只显示前 10 个
for i, entry in enumerate(call_log[:10], 1):
    ...

# 只显示特定层
first_layer_calls = [
    entry for entry in call_log 
    if 'layers.0.' in entry['name']  # 只看第 0 层
]
```

### Q4: 如何保存追踪结果？

A: 在脚本末尾添加：

```python
# 保存到文件
with open('trace_result.txt', 'w') as f:
    for entry in call_log:
        f.write(f"{entry}\n")
```

---

## 🎓 学习建议

1. **第一次运行**：使用 `test_simple_trace.py`
   - 理解基本的执行流程
   - 看到每个模块的调用次数

2. **深入理解**：使用 `test_cuda_profiler.py`
   - 看到实际的 CUDA kernel 调用
   - 理解哪些操作最耗时

3. **集成新算子**：参考这些脚本
   - 确认你的算子被正确调用
   - 对比性能（调用次数、耗时）

---

## 📚 相关文档

- [vLLM 架构文档](../docs/paged_attention_v2_analysis.md)
- [CoW 代码位置指南](../docs/cow_code_locations.md)

---

**提示**：这些脚本会在推理时添加 hook，可能会略微影响性能。生产环境请移除追踪代码。
