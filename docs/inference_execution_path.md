# vLLM V1 推理执行路径分析

## 🔍 你的问题：在 `LLMEngine.step()` 找不到推理代码

你看到的代码：
```python
def step(self) -> list[RequestOutput | PoolingRequestOutput]:
    if self.should_execute_dummy_batch:
        self.should_execute_dummy_batch = False
        self.engine_core.execute_dummy_batch()
        return []
    
    # 1) Get EngineCoreOutput from the EngineCore.
    outputs = self.engine_core.get_output()  # ← 推理在这里！
    
    # 2) Process EngineCoreOutputs.
    processed_outputs = self.output_processor.process_outputs(...)
    
    # 3) Abort any reqs that finished
    self.engine_core.abort_requests(...)
    
    return processed_outputs.request_outputs
```

**关键理解**：推理代码不在 `llm_engine.py`，而在 `engine_core` 中！

---

## 📊 完整的调用链

```
你的代码：llm.generate(prompts)
    ↓
vllm/entrypoints/llm.py: LLM.generate()
    ↓
    while engine.has_unfinished_requests():
        outputs = engine.step()  ← 你看到的这个
    ↓
vllm/v1/engine/llm_engine.py: LLMEngine.step()
    ↓
    outputs = self.engine_core.get_output()  ← 关键！
    ↓
vllm/v1/engine/core_client.py: EngineCoreClient.get_output()
    ↓
vllm/v1/engine/core.py: EngineCore.step()  ← 真正的推理在这里！
    ↓
    self._schedule()           # 调度请求
    self._execute_model()      # 🔥 执行模型（调用算子）
    ↓
vllm/v1/executor/gpu_executor.py: GPUExecutor.execute_model()
    ↓
vllm/v1/worker/gpu_worker.py: GPUWorker.execute_model()
    ↓
vllm/v1/worker/gpu_model_runner.py: GPUModelRunner.execute_model()
    ↓
    output = self.model(...)  ← 调用 Transformer 模型
    ↓
vllm/model_executor/models/qwen2.py: Qwen2ForCausalLM.forward()
    ↓
    for layer in self.layers:
        hidden_states = layer(hidden_states, ...)  ← 逐层计算
    ↓
vllm/model_executor/models/qwen2.py: Qwen2DecoderLayer.forward()
    ↓
    # RMSNorm
    hidden_states = self.input_layernorm(hidden_states)
    # Attention (调用 PagedAttention)
    hidden_states = self.self_attn(hidden_states, kv_cache, ...)
    # MLP
    hidden_states = self.mlp(hidden_states)
    ↓
vllm/model_executor/layers/attention.py: Attention.forward()
    ↓
vllm/attention/backends/flash_attn.py: FlashAttentionImpl.forward()
    ↓
    torch.ops.vllm.paged_attention_v2(...)  ← 🔥 调用 CUDA kernel
    ↓
csrc/attention/paged_attention_v2.cu: paged_attention_v2()
    ↓
    paged_attention_v2_kernel<<<>>>()        ← GPU 计算
    paged_attention_v2_reduce_kernel<<<>>>() ← GPU 归约
```

---

## 🎯 关键文件位置

### 1. EngineCore - 真正的推理逻辑

```python
# 文件：vllm/v1/engine/core.py

class EngineCore:
    """核心推理引擎"""
    
    def step(self) -> EngineCoreOutput:
        """单步推理（这里才是推理代码！）"""
        
        # 🔍 调度：选择要执行的请求
        scheduler_output = self._schedule()
        
        # 🔥 执行：调用模型前向传播
        model_output = self._execute_model(scheduler_output)
        
        # 📤 处理输出
        return self._process_model_output(model_output)
    
    def _schedule(self) -> SchedulerOutput:
        """调度器：决定哪些请求执行"""
        # 选择 batch
        # 分配 KV cache blocks
        # 更新请求状态
        pass
    
    def _execute_model(self, scheduler_output) -> ModelOutput:
        """🔥 执行模型（调用所有算子）"""
        
        # 准备输入
        model_input = self._prepare_model_input(scheduler_output)
        
        # 🔥 调用 Executor
        output = self.model_executor.execute_model(
            execute_model_req=model_input
        )
        
        return output
```

**位置**：`vllm/v1/engine/core.py`

### 2. ModelExecutor - 执行模型

```python
# 文件：vllm/v1/executor/gpu_executor.py

class GPUExecutor:
    def execute_model(self, execute_model_req):
        """执行模型前向传播"""
        
        # 🔥 调用 Worker
        output = self.driver_worker.execute_model(
            execute_model_req=execute_model_req
        )
        
        return output
```

### 3. GPUWorker - 实际执行

```python
# 文件：vllm/v1/worker/gpu_worker.py

class GPUWorker:
    def execute_model(self, execute_model_req):
        """在 GPU 上执行模型"""
        
        # 🔥 调用 ModelRunner
        output = self.model_runner.execute_model(
            model_input=execute_model_req.model_input,
            kv_caches=self.kv_caches,
        )
        
        return output
```

### 4. ModelRunner - 调用模型

```python
# 文件：vllm/v1/worker/gpu_model_runner.py

class GPUModelRunner:
    def execute_model(self, model_input, kv_caches):
        """执行模型前向传播"""
        
        # 准备输入张量
        input_ids = model_input.input_ids
        positions = model_input.positions
        
        # 🔥 调用模型
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=model_input.attn_metadata,
        )
        
        # Logits 计算
        logits = self.model.compute_logits(hidden_states, ...)
        
        return logits
```

### 5. Model - Transformer 层

```python
# 文件：vllm/model_executor/models/qwen2.py

class Qwen2ForCausalLM(nn.Module):
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: list[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ):
        # Embedding
        hidden_states = self.embed_tokens(input_ids)
        
        # 🔥 逐层计算
        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                positions=positions,
                hidden_states=hidden_states,
                kv_cache=kv_caches[i],
                attn_metadata=attn_metadata,
            )
        
        # Final norm
        hidden_states = self.norm(hidden_states)
        
        return hidden_states
```

### 6. DecoderLayer - 单层计算

```python
# 文件：vllm/model_executor/models/qwen2.py

class Qwen2DecoderLayer(nn.Module):
    def forward(self, positions, hidden_states, kv_cache, attn_metadata):
        # 🔹 Pre-Attention RMSNorm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # 🔥 Self-Attention（调用 PagedAttention）
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )
        hidden_states = residual + hidden_states
        
        # 🔹 Post-Attention RMSNorm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # 🔹 MLP
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states
```

---

## 🔥 算子调用的具体位置

### 1. RMSNorm 算子

```python
# 文件：vllm/model_executor/layers/layernorm.py

class RMSNorm(nn.Module):
    def forward(self, x, residual=None):
        if residual is not None:
            # 🔥 调用 CUDA kernel
            x = torch.ops.vllm.fused_add_rms_norm(
                x, residual, self.weight, self.variance_epsilon
            )
            # → csrc/ops/layernorm.cu: fused_add_rms_norm_kernel<<<>>>
        else:
            # PyTorch 原生实现
            pass
        return x
```

**CUDA 文件**：`csrc/ops/layernorm.cu`

### 2. Rotary Embedding 算子

```python
# 文件：vllm/model_executor/layers/rotary_embedding.py

class RotaryEmbedding(nn.Module):
    def forward(self, positions, query, key):
        # 🔥 调用 CUDA kernel
        torch.ops.vllm.rotary_embedding(
            positions, query, key, self.head_size, ...
        )
        # → csrc/ops/rotary_embedding.cu: rotary_embedding_kernel<<<>>>
        return query, key
```

**CUDA 文件**：`csrc/ops/rotary_embedding.cu`

### 3. PagedAttention 算子

```python
# 文件：vllm/attention/ops/paged_attn.py

def paged_attention_v2(...):
    """PagedAttention V2 算子"""
    
    # 🔥 调用 CUDA kernels
    torch.ops.vllm.paged_attention_v2(
        out, exp_sums, max_logits, tmp_out,
        query, key_cache, value_cache,
        num_kv_heads, scale, block_tables, seq_lens,
        ...
    )
    # → csrc/attention/paged_attention_v2.cu:
    #     - paged_attention_v2_kernel<<<>>>
    #     - paged_attention_v2_reduce_kernel<<<>>>
```

**CUDA 文件**：`csrc/attention/paged_attention_v2.cu`

### 4. SiLU 激活函数算子

```python
# 文件：vllm/model_executor/layers/activation.py

class SiluAndMul(nn.Module):
    def forward(self, x):
        # 🔥 调用 CUDA kernel
        torch.ops.vllm.silu_and_mul(out, x)
        # → csrc/ops/activation.cu: silu_and_mul_kernel<<<>>>
        return out
```

**CUDA 文件**：`csrc/ops/activation.cu`

---

## 📝 如何追踪推理路径？

### 方法 1：添加日志到关键文件

我已经在 `llm_engine.py` 中添加了日志。现在你需要在其他关键文件中添加：

#### 在 EngineCore 中添加日志

```python
# 文件：vllm/v1/engine/core.py

class EngineCore:
    def step(self):
        logger.info("🔹 [EngineCore.step] 开始")
        
        # 调度
        logger.info("   → 调度请求...")
        scheduler_output = self._schedule()
        logger.info("   → 选中 %d 个请求", len(scheduler_output.scheduled_requests))
        
        # 执行模型
        logger.info("   → 🔥 执行模型...")
        model_output = self._execute_model(scheduler_output)
        logger.info("   → ✅ 模型执行完成")
        
        return self._process_model_output(model_output)
```

#### 在 ModelRunner 中添加日志

```python
# 文件：vllm/v1/worker/gpu_model_runner.py

class GPUModelRunner:
    def execute_model(self, model_input, kv_caches):
        logger.info("🔥 [ModelRunner] 执行模型前向传播")
        logger.info("   → input_ids shape: %s", model_input.input_ids.shape)
        
        # 调用模型
        hidden_states = self.model(
            input_ids=model_input.input_ids,
            positions=model_input.positions,
            kv_caches=kv_caches,
            attn_metadata=model_input.attn_metadata,
        )
        
        logger.info("   → ✅ hidden_states shape: %s", hidden_states.shape)
        return logits
```

### 方法 2：使用我创建的追踪脚本

运行：
```bash
python test_simple_trace.py
```

这会自动追踪所有模块的调用。

---

## 🎯 总结

### 推理代码的位置

| 层次 | 文件 | 职责 |
|------|------|------|
| **1. 入口** | `vllm/entrypoints/llm.py` | 用户调用 `llm.generate()` |
| **2. 引擎** | `vllm/v1/engine/llm_engine.py` | 循环调用 `step()` |
| **3. 核心** | `vllm/v1/engine/core.py` | **真正的推理逻辑（调度+执行）** |
| **4. 执行器** | `vllm/v1/executor/gpu_executor.py` | 分发任务到 Worker |
| **5. Worker** | `vllm/v1/worker/gpu_worker.py` | GPU 上执行 |
| **6. Runner** | `vllm/v1/worker/gpu_model_runner.py` | 准备输入，调用模型 |
| **7. 模型** | `vllm/model_executor/models/qwen2.py` | Transformer 层 |
| **8. Layers** | `vllm/model_executor/layers/` | 各种算子（RMSNorm、Attention、MLP）|
| **9. CUDA** | `csrc/attention/`, `csrc/ops/` | **GPU Kernels** |

### 关键理解

1. **`llm_engine.py` 只是外壳**
   - 真正的推理在 `engine_core` 中

2. **推理路径**：
   ```
   llm_engine.step()
     → engine_core.get_output()
       → EngineCore.step()
         → EngineCore._execute_model()
           → GPUExecutor.execute_model()
             → GPUWorker.execute_model()
               → ModelRunner.execute_model()
                 → Model.forward()
                   → Layer.forward()
                     → Attention/MLP/RMSNorm
                       → CUDA Kernels
   ```

3. **如何找到算子调用**？
   - 在 `vllm/model_executor/layers/` 中找到对应的 Layer
   - 看 `forward()` 方法
   - 找到 `torch.ops.vllm.*` 的调用
   - 这些就是 CUDA kernel 的入口

### 下一步

如果你想添加 SonicMoE，需要：
1. 在 `vllm/model_executor/layers/sonic_moe.py` 创建 Layer
2. 在模型中使用（如 `Qwen2DecoderLayer`）
3. Layer 内部调用你的 CUDA kernel

运行我的追踪脚本，你会看到完整的调用路径！🎯
