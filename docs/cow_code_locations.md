# vLLM Copy-on-Write (CoW) 代码位置指南

## 📍 关键文件位置

### 1. Block Allocator（核心 CoW 实现）

```python
# vllm/v1/core/kv_cache_coordinator/block_allocator.py
# 或
# vllm/core/block/block_allocator.py
```

**关键类和方法**：
- `BlockAllocator` 或 `CachedBlockAllocator`
- `get_ref_count(block_id)` - 获取引用计数
- `fork(src_block, dst_block)` - 复制 block（CoW）
- `free(block_id)` - 释放 block，减少引用计数

### 2. KV Cache Manager

```python
# vllm/v1/core/kv_cache_manager.py
```

**相关方法**：
- `fork_cache_blocks()` - fork KV cache blocks
- `allocate_slots()` - 分配 slots 时可能触发 CoW

### 3. Prefix Caching

```python
# vllm/v1/core/kv_cache_coordinator/single_node.py
# vllm/v1/core/kv_cache_coordinator/prefix_caching.py
```

**相关功能**：
- Prefix caching 重度依赖 CoW
- 多个请求共享相同的 prefix blocks

---

## 🔍 查找方法

### 方法 1：直接搜索

```bash
cd c:\Users\water\workspace\vllm-dynamic-sparsity

# 搜索 get_ref_count
grep -r "get_ref_count" vllm/

# 搜索 copy_on_write 或 CoW
grep -r "copy_on_write\|CoW" vllm/

# 搜索 fork 方法（block fork）
grep -r "def fork" vllm/v1/core/
```

### 方法 2：查看具体文件

主要文件路径（按优先级）：

1. **Block Allocator**:
   - `vllm/v1/core/kv_cache_coordinator/block_allocator.py`
   - `vllm/core/block_manager_v2.py`
   - `vllm/core/block/block_allocator.py`

2. **KV Cache Coordinator**:
   - `vllm/v1/core/kv_cache_coordinator/single_node.py`
   - `vllm/v1/core/kv_cache_coordinator/__init__.py`

3. **Block Manager**:
   - `vllm/core/block_manager.py`
   - `vllm/core/block_manager_v2.py`

---

## 📝 预期的 CoW 代码模式

### 示例 1：引用计数

```python
class BlockAllocator:
    def __init__(self):
        self.ref_counts = {}  # block_id -> ref_count
    
    def get_ref_count(self, block_id: int) -> int:
        """获取 block 的引用计数"""
        return self.ref_counts.get(block_id, 0)
    
    def incr_ref_count(self, block_id: int) -> None:
        """增加引用计数（共享 block）"""
        if block_id not in self.ref_counts:
            self.ref_counts[block_id] = 0
        self.ref_counts[block_id] += 1
    
    def decr_ref_count(self, block_id: int) -> None:
        """减少引用计数（释放时）"""
        self.ref_counts[block_id] -= 1
        if self.ref_counts[block_id] == 0:
            # 真正释放 block
            self._free_block(block_id)
```

### 示例 2：Copy-on-Write

```python
def fork_cache_blocks(
    self,
    src_request_id: str,
    dst_request_id: str,
    num_blocks: int
) -> None:
    """
    Fork cache blocks（Copy-on-Write）
    
    不实际复制数据，只增加引用计数。
    当需要修改时才真正复制。
    """
    src_blocks = self.get_blocks(src_request_id)
    
    for i in range(num_blocks):
        block_id = src_blocks[i]
        
        # 增加引用计数（共享）
        self.allocator.incr_ref_count(block_id)
        
        # 目标请求也指向同一个 block
        self.assign_block(dst_request_id, i, block_id)

def write_to_cache(self, request_id: str, block_idx: int, data):
    """
    写入 cache 时检查是否需要 CoW
    """
    block_id = self.get_block(request_id, block_idx)
    
    # 检查引用计数
    if self.allocator.get_ref_count(block_id) > 1:
        # 有其他请求共享，需要 Copy-on-Write
        new_block_id = self.allocator.allocate_block()
        
        # 复制数据
        self._copy_block(src=block_id, dst=new_block_id)
        
        # 减少原 block 的引用计数
        self.allocator.decr_ref_count(block_id)
        
        # 更新指向新 block
        self.assign_block(request_id, block_idx, new_block_id)
        
        block_id = new_block_id
    
    # 写入数据
    self._write_data(block_id, data)
```

### 示例 3：Prefix Caching 使用 CoW

```python
def allocate_with_prefix_caching(
    self,
    request_id: str,
    prompt_tokens: List[int]
) -> None:
    """
    使用 prefix caching 分配 blocks
    """
    # 1. 查找匹配的 prefix
    matched_blocks, matched_len = self._find_prefix_match(prompt_tokens)
    
    if matched_len > 0:
        # 2. 共享匹配的 prefix blocks（CoW）
        for i, block_id in enumerate(matched_blocks):
            # 增加引用计数（不复制数据）
            self.allocator.incr_ref_count(block_id)
            self.assign_block(request_id, i, block_id)
    
    # 3. 为剩余 tokens 分配新 blocks
    remaining_tokens = prompt_tokens[matched_len:]
    for token in remaining_tokens:
        new_block = self.allocator.allocate_block()
        self.assign_block(request_id, len(self.get_blocks(request_id)), new_block)
```

---

## 🎯 在你的代码库中查找

### 步骤 1：查找 BlockAllocator

```python
# 查找文件
find vllm -name "*allocator*.py" -o -name "*block*.py"

# 可能的文件：
# - vllm/v1/core/kv_cache_coordinator/block_allocator.py
# - vllm/core/block_manager.py
# - vllm/core/block/block_allocator.py
```

### 步骤 2：查看 KVCacheManager

```python
# 文件位置
vllm/v1/core/kv_cache_manager.py

# 查找方法
grep -n "get_ref_count\|fork\|cow" vllm/v1/core/kv_cache_manager.py
```

### 步骤 3：查看 Coordinator

```python
# 文件位置
vllm/v1/core/kv_cache_coordinator/

# 列出所有文件
ls vllm/v1/core/kv_cache_coordinator/
```

---

## 💡 具体查找命令

在你的项目根目录运行：

```bash
# 1. 查找 get_ref_count 方法定义
grep -rn "def get_ref_count" vllm/

# 2. 查找 get_ref_count 方法调用
grep -rn "\.get_ref_count\(" vllm/

# 3. 查找 fork 相关代码
grep -rn "def fork\|fork_cache\|fork_blocks" vllm/

# 4. 查找 ref_count 相关
grep -rn "ref_count\|reference_count" vllm/

# 5. 查找 Copy-on-Write 注释
grep -rn "copy.*on.*write\|CoW" vllm/ -i

# 6. 列出可能包含 CoW 代码的文件
find vllm -name "*.py" | xargs grep -l "ref_count\|get_ref_count"
```

---

## 📊 预期输出示例

运行上述命令后，你应该能看到类似：

```
vllm/v1/core/kv_cache_coordinator/block_allocator.py:123: def get_ref_count(self, block_id: int) -> int:
vllm/v1/core/kv_cache_coordinator/block_allocator.py:145: def incr_ref_count(self, block_id: int) -> None:
vllm/v1/core/kv_cache_manager.py:234:     if self.allocator.get_ref_count(block_id) > 1:
vllm/v1/core/kv_cache_manager.py:567: def fork_cache_blocks(self, src_req, dst_req):
```

---

## 🔗 相关概念

### Copy-on-Write 的优势

1. **内存节省**：多个请求共享相同的 prefix
2. **性能提升**：避免不必要的内存复制
3. **Prefix Caching**：重用计算结果

### 引用计数的作用

- `ref_count = 1`：只有一个请求使用，可以直接修改
- `ref_count > 1`：多个请求共享，修改前需要 CoW
- `ref_count = 0`：没有请求使用，可以释放

### CoW 触发时机

1. **Write 时检查**：写入 KV cache 前检查 ref_count
2. **Fork 时增加**：fork cache blocks 时增加 ref_count
3. **Free 时减少**：释放 blocks 时减少 ref_count

---

**建议**：直接运行上面的 grep 命令，找到具体的文件和行号！
