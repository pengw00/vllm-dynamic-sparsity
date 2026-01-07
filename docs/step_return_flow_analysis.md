# EngineCore.step() 执行完后的完整流程

## 🎯 你的问题

**问**：`EngineCore.step()` 执行到这里后呢？

```python
engine_core_outputs = self.scheduler.update_from_output(
    scheduler_output, model_output
)
return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0
```

**答**：返回到 `InprocClient.get_output()`，然后一层层返回给用户代码。

---

## 📊 完整的返回路径

```
EngineCore.step() 返回
    ↓
    return (engine_core_outputs, model_executed)
    ↓
InprocClient.get_output() 接收
    ↓
    outputs, model_executed = self.engine_core.step_fn()
    ↓
    return outputs.get(0) or EngineCoreOutputs()
    ↓
LLMEngine.step() 接收
    ↓
    outputs = self.engine_core.get_output()
    ↓
    processed_outputs = self.output_processor.process_outputs(outputs)
    ↓
    return processed_outputs.request_outputs
    ↓
LLM.generate() 接收
    ↓
    while engine.has_unfinished_requests():
        outputs = engine.step()  ← 得到这一步的输出
        for output in outputs:
            if output.finished:
                final_outputs.append(output)
    ↓
返回给用户
    ↓
你的代码
    outputs = llm.generate(prompts)
    print(outputs[0].outputs[0].text)  ← 最终结果
```

---

## 🔍 详细分析每一步

### Step 1: EngineCore.step() 返回

```python
# 文件：vllm/v1/engine/core.py

def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
    # ... 执行推理 ...
    
    # 最后一行：返回结果
    return engine_core_outputs, model_executed
    #      ↑                    ↑
    #      |                    是否执行了模型（bool）
    #      |
    #      输出字典：{wave_id: EngineCoreOutputs}

# engine_core_outputs 的结构：
# {
#     0: EngineCoreOutputs(
#         outputs=[
#             EngineCoreOutput(
#                 request_id="req_123",
#                 new_token_ids=[123, 456],  # 新生成的 token
#                 finish_reason=None,         # 如果完成了会有值
#             ),
#             ...
#         ]
#     )
# }
```

---

### Step 2: InprocClient.get_output() 接收并处理

```python
# 文件：vllm/v1/engine/core_client.py

def get_output(self) -> EngineCoreOutputs:
    logger.info("🔸 [InprocClient.get_output] 调用 EngineCore.step_fn()")
    
    # 接收 EngineCore.step() 的返回值
    outputs, model_executed = self.engine_core.step_fn()
    #                         ↑
    #                         self.engine_core.step_fn 指向 EngineCore.step
    
    logger.info("   → model_executed: %s", model_executed)
    logger.info("   → outputs 类型: %s", type(outputs))
    
    # 后处理
    self.engine_core.post_step(model_executed=model_executed)
    
    # 提取 wave 0 的输出（大多数情况只有一个 wave）
    return outputs and outputs.get(0) or EngineCoreOutputs()
    #      ↑
    #      如果 outputs 不为空，获取 wave 0 的输出
    #      否则返回空的 EngineCoreOutputs
```

**post_step() 做什么？**

```python
def post_step(self, model_executed: bool) -> None:
    """后处理步骤"""
    
    # 如果使用 speculative decoding，更新 draft token ids
    if not self.async_scheduling and self.use_spec_decode and model_executed:
        draft_token_ids = self.model_executor.take_draft_token_ids()
        if draft_token_ids is not None:
            self.scheduler.update_draft_token_ids(draft_token_ids)
    
    # 对于你的简单场景，这里基本什么都不做
```

---

### Step 3: LLMEngine.step() 接收并转换

```python
# 文件：vllm/v1/engine/llm_engine.py

def step(self) -> list[RequestOutput | PoolingRequestOutput]:
    logger.info("="*80)
    logger.info("🔹 [LLMEngine.step] 开始新的推理 step")
    logger.info("="*80)
    
    # Step 1: 从 EngineCore 获取输出
    logger.info("📥 [Step 1] 从 EngineCore 获取输出...")
    outputs = self.engine_core.get_output()
    #         ↑
    #         InprocClient.get_output() 返回的结果
    
    logger.info("✅ [Step 1] 获取到 outputs")
    logger.info("   → outputs 类型: %s", type(outputs).__name__)
    
    # Step 2: 处理输出 - 转换为用户友好的格式
    logger.info("\n📊 [Step 2] 处理输出...")
    logger.info("   → 调用 output_processor.process_outputs()")
    
    processed_outputs = self.output_processor.process_outputs(
        outputs.outputs,  # EngineCoreOutput 列表
        engine_core_timestamp=outputs.timestamp,
        iteration_stats=iteration_stats,
    )
    
    logger.info("✅ [Step 2] 输出处理完成")
    logger.info("   → 返回的 RequestOutput 数量: %d", 
               len(processed_outputs.request_outputs))
    
    # Step 3: 中止已完成的请求
    logger.info("\n🗑️  [Step 3] 处理中止请求")
    self.engine_core.abort_requests(processed_outputs.reqs_to_abort)
    
    # Step 4: 返回用户可见的输出
    logger.info("\n✅ 返回 RequestOutput 列表")
    return processed_outputs.request_outputs
    #      ↑
    #      这是用户友好的格式
    #      list[RequestOutput]
```

**OutputProcessor.process_outputs() 做什么？**

```python
# 文件：vllm/v1/engine/output_processor.py

def process_outputs(
    self,
    outputs: list[EngineCoreOutput],
    engine_core_timestamp: float,
    iteration_stats: IterationStats | None,
) -> ProcessedOutputs:
    """
    转换 EngineCoreOutput -> RequestOutput
    
    EngineCoreOutput (内部格式):
        - request_id: str
        - new_token_ids: list[int]
        - finish_reason: FinishReason | None
    
    RequestOutput (用户格式):
        - request_id: str
        - prompt: str
        - prompt_token_ids: list[int]
        - outputs: list[CompletionOutput]
            - text: str  ← 解码后的文本！
            - token_ids: list[int]
            - finish_reason: str | None
    """
    
    request_outputs = []
    
    for output in outputs:
        # 获取请求状态
        request_state = self.request_states[output.request_id]
        
        # 累积新生成的 tokens
        request_state.token_ids.extend(output.new_token_ids)
        
        # 🔥 解码 tokens 为文本
        text = self.tokenizer.decode(
            request_state.token_ids,
            skip_special_tokens=True
        )
        
        # 创建 RequestOutput
        request_output = RequestOutput(
            request_id=output.request_id,
            prompt=request_state.prompt_text,
            prompt_token_ids=request_state.prompt_token_ids,
            outputs=[
                CompletionOutput(
                    text=text,  ← 这是用户看到的文本！
                    token_ids=request_state.token_ids,
                    finish_reason=output.finish_reason,
                )
            ],
            finished=(output.finish_reason is not None),
        )
        
        request_outputs.append(request_output)
    
    return ProcessedOutputs(request_outputs=request_outputs)
```

---

### Step 4: LLM.generate() 循环接收

```python
# 文件：vllm/entrypoints/llm.py

def generate(
    self,
    prompts: list[str],
    sampling_params: SamplingParams,
) -> list[RequestOutput]:
    """用户调用的生成方法"""
    
    logger.info("🚀 [LLM.generate] 开始生成")
    logger.info("   Prompt: %s", prompts[0])
    logger.info("   Max tokens: %d", sampling_params.max_tokens)
    
    # 添加请求到引擎
    for prompt in prompts:
        request_id = f"req_{uuid.uuid4()}"
        self.engine.add_request(
            request_id=request_id,
            prompt=prompt,
            params=sampling_params,
        )
    
    # 🔥 核心循环：不断调用 engine.step() 直到所有请求完成
    final_outputs = []
    step_count = 0
    
    logger.info("\n⚡ 开始生成循环")
    
    while self.engine.has_unfinished_requests():
        step_count += 1
        logger.info("\n--- Step %d ---", step_count)
        
        # 执行一步推理
        outputs = self.engine.step()
        #         ↑
        #         这返回 list[RequestOutput]
        #         包含这一步所有请求的输出
        
        logger.info("   → 本步输出数量: %d", len(outputs))
        
        # 处理每个输出
        for output in outputs:
            logger.info("   → Request %s:", output.request_id)
            logger.info("     • 当前文本: %s", output.outputs[0].text)
            logger.info("     • 已生成 tokens: %d", len(output.outputs[0].token_ids))
            logger.info("     • 是否完成: %s", output.finished)
            
            if output.finished:
                logger.info("     ✅ 请求完成！")
                final_outputs.append(output)
    
    logger.info("\n✅ [LLM.generate] 所有请求完成")
    logger.info("   总步数: %d", step_count)
    
    return final_outputs
```

---

### Step 5: 返回给用户

```python
# 你的代码
outputs = llm.generate(prompts, sampling_params)
#         ↑
#         这里接收到 list[RequestOutput]

# 访问结果
for output in outputs:
    print(f"Generated: {output.outputs[0].text}")
    #                   ↑
    #                   这是解码后的完整文本
```

---

## 🔄 单个 Token 生成的完整数据流

```
Step 1: 用户调用
    llm.generate(["Tell me a joke"])
    ↓

Step 2: 添加请求
    engine.add_request(request_id="req_123", prompt="Tell me a joke")
    ↓

Step 3: 循环开始
    while engine.has_unfinished_requests():
    ↓

Step 4: 执行推理（第一次）
    outputs = engine.step()
        ↓
    LLMEngine.step()
        ↓
    engine_core.get_output()
        ↓
    InprocClient.get_output()
        ↓
    EngineCore.step()
        ├─ scheduler.schedule() → 调度 prompt tokens
        ├─ model_executor.execute_model() → 前向传播
        │    ├─ Model.forward() → Transformer 计算
        │    └─ 返回 logits: [batch, vocab_size]
        ├─ sample_tokens() → 从 logits 采样下一个 token
        │    └─ 得到 token_id = 1234
        └─ scheduler.update_from_output() → 更新状态
             └─ 返回 EngineCoreOutput(
                    request_id="req_123",
                    new_token_ids=[1234],  ← 新 token
                    finish_reason=None
                )
    ↓
    output_processor.process_outputs()
        ├─ 累积 tokens: [1234]
        ├─ 解码: tokenizer.decode([1234]) = "Why"
        └─ 返回 RequestOutput(
               text="Why",  ← 当前文本
               finished=False
           )
    ↓
    返回给 generate() 循环
    ↓

Step 5: 继续循环（第二次）
    outputs = engine.step()
        ... 同样的流程 ...
        新 token_id = 5678
        累积 tokens: [1234, 5678]
        解码: "Why do"
        返回 RequestOutput(text="Why do", ...)
    ↓

... 重复多次 ...

Step N: 最后一次（遇到停止条件）
    outputs = engine.step()
        新 token_id = EOS_TOKEN
        finish_reason = FinishReason.STOP
        返回 RequestOutput(
            text="Why do software engineers...",  ← 完整文本
            finished=True  ← 标记完成
        )
    ↓

Step N+1: 退出循环
    has_unfinished_requests() → False
    ↓
    返回 final_outputs
    ↓

用户接收结果
    outputs = [RequestOutput(text="Why do software engineers...")]
```

---

## 📊 数据类型转换链

```
EngineCore 内部格式：
EngineCoreOutput
├── request_id: str
├── new_token_ids: list[int]  ← 原始 token IDs
└── finish_reason: FinishReason | None
    ↓
    OutputProcessor 转换
    ↓
用户可见格式：
RequestOutput
├── request_id: str
├── prompt: str
├── prompt_token_ids: list[int]
└── outputs: list[CompletionOutput]
    └── CompletionOutput
        ├── text: str  ← 解码后的文本！
        ├── token_ids: list[int]
        └── finish_reason: str | None
```

---

## 🎯 关键理解

### 1. 返回路径是逐层返回的

```python
EngineCore.step()
    return engine_core_outputs, model_executed
    ↓
InprocClient.get_output()
    outputs, model_executed = self.engine_core.step_fn()
    return outputs.get(0)
    ↓
LLMEngine.step()
    outputs = self.engine_core.get_output()
    processed = self.output_processor.process_outputs(outputs)
    return processed.request_outputs
    ↓
LLM.generate()
    outputs = self.engine.step()
    # 收集所有完成的输出
    ↓
你的代码
    outputs = llm.generate(...)
```

### 2. 每一步都在转换数据格式

```
EngineCore.step()
    ↓ EngineCoreOutput (内部格式，包含 token IDs)
InprocClient.get_output()
    ↓ EngineCoreOutputs (包装格式)
LLMEngine.step()
    ↓ OutputProcessor 转换
    ↓ RequestOutput (用户格式，包含解码后的文本)
LLM.generate()
    ↓ list[RequestOutput]
你的代码
    ↓ 最终结果
```

### 3. 是一个循环过程

```python
while engine.has_unfinished_requests():
    # 每次循环生成一个或多个 token
    outputs = engine.step()
    
    # 如果请求完成了，收集结果
    for output in outputs:
        if output.finished:
            final_outputs.append(output)

# 所有请求完成后退出循环
return final_outputs
```

---

## 📝 总结

**`EngineCore.step()` 执行完后的流程**：

1. **返回结果** → `InprocClient.get_output()`
2. **提取输出** → 获取 wave 0 的 `EngineCoreOutputs`
3. **后处理** → `post_step()` 做清理工作
4. **返回** → `LLMEngine.step()`
5. **转换格式** → `OutputProcessor` 解码 tokens 为文本
6. **返回** → `LLM.generate()` 循环
7. **检查是否完成** → 如果完成，退出循环
8. **返回给用户** → 最终的 `list[RequestOutput]`

**关键点**：
- ✅ 每次 `step()` 生成 1 个或多个 tokens
- ✅ 结果逐层返回，每层做不同的处理
- ✅ OutputProcessor 负责解码 tokens 为文本
- ✅ generate() 循环直到所有请求完成

现在你清楚 `step()` 执行完后的完整流程了！🎯
