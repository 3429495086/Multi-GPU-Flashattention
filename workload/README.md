mask_value()：说明为什么 mask 规则要集中在一个地方
build_tasks()：说明为什么只输出非空 block
build_row_tasks()：说明它是把 block 聚合成 row-level task
assign_lpt_scan()：说明为什么要排序但还要保留原始下标
assign_lpt()：说明它现在是 idealized baseline
rebalance_move_once() / rebalance_swap_once()：说明它们分别在修什么
assign_row_tasks_lpt()：说明 row-level LPT 的调度单位变了
partition_rows_contiguous()：说明 DP 状态、切点和回溯分别在干什么

可以。按你现在这套代码，我给你按“模块”总结最关键的变量名，各自是干什么的。

**`workload_input`**
- `seq_q`：Q 序列总长度。
- `seq_k`：K 序列总长度。
- `block_q`：Q 方向每个 block 的长度。
- `block_k`：K 方向每个 block 的长度。

它的作用是：决定 attention 被切成多少个 block。

**`mask_desc`**
- `type`：mask 类型。你现在主要是 `MASK_CAUSAL`。

它的作用是：定义哪些 `(q, k)` 位置有效。

**`cost_model`**
- `alpha`：固定开销。
- `beta`：每个有效元素对应的线性代价。

它的作用是：把 `active` 转成 `cost`。

---

**`block_task`**
这是 block-level workload 的基本单位。

- `row_idx`：这个 block 在 block 网格里的行号。
- `col_idx`：这个 block 在 block 网格里的列号。
- `q_begin`, `q_end`：这个 block 覆盖的 Q 范围，左闭右开。
- `k_begin`, `k_end`：这个 block 覆盖的 K 范围，左闭右开。
- `active`：这个 block 里有效元素个数。
- `cost`：这个 block 的估计计算代价。
- `total`：这个 block 的元素总数。
- `density`：有效比例，`active / total`。

它的作用是：描述一个非空 attention block 的几何范围和工作量。

---

**`row_task`**
这是 row-level / shard-level 调度的基础单位，是把同一行的多个 `block_task` 聚合后的结果。

- `row_idx`：第几行 block-row。
- `num_blocks`：这一行有多少个非空 block。
- `num_cols_needed`：这一行实际需要多少列 KV block。causal 下等于“最右非空列 + 1”。
- `row_cost`：这一行所有 block 的计算 cost 总和。
- `comm_cost`：这一行的通信代价。你现在还是 `0`。
- `total_cost`：总成本，`row_cost + comm_cost`。

它的作用是：把细粒度 block workload 升级成更贴近 Ring 的行级任务。

---

**`gpu_load`**
这是 scheduler 里每张 GPU 当前的负载状态。

- `gpu_id`：GPU 编号。
- `task_count`：当前分到了多少个任务。
- `total_cost`：当前这张 GPU 的总代价。

它的作用是：让 LPT、move、swap 能比较谁最轻、谁最重。

---

**`task_to_gpu[]`**
- 下标是 task 编号。
- 值是这个 task 被分到哪张 GPU。

比如：
```c
task_to_gpu[5] = 2;
```
表示第 5 个 `block_task` 在 GPU 2 上。

它的作用是：从“只知道每张卡多重”变成“知道每个任务具体去哪了”。  
move / swap repair 都靠它。

---

**`row_to_gpu[]`**
- 下标是 row 编号。
- 值是这个 `row_task` 被分到哪张 GPU。

它的作用和 `task_to_gpu[]` 一样，只不过对象变成了 row-level task。

---

**`indexed_task`**
这是 scheduler 内部排序时用的辅助结构。

- `original_idx`：原始任务编号。
- `cost`：排序时用的代价。

它的作用是：排序后仍然知道“原来是谁”，这样 `task_to_gpu[idx]` 才能填对。

---

**`row_shard`**
这是连续分段 DP 的输出结构。

- `gpu_id`：这段属于哪张 GPU。
- `row_begin`：连续段起点。
- `row_end`：连续段终点，左闭右开。
- `total_cost`：这个 shard 的总代价。

它的作用是：表达“连续 shard”方案的最终切分结果。

---

**DP 里的关键变量**
在 `partition_rows_contiguous()` 里：

- `prefix[i]`：前 `i` 行的总代价前缀和。
  用来快速算连续段 `[k, i)` 的 cost。

- `dp[g][i]`：前 `i` 行分给 `g` 张 GPU 时，最优的最小 makespan。
  这是 DP 的主状态。

- `cut[g][i]`：达到 `dp[g][i]` 时，最后一段从哪一行开始。
  用来回溯最终切分结果。

- `k`：当前枚举的最后一个分割点候选。
  也就是最后一段 `[k, i)` 的起点。

---

**schedule summary 里的统计量**
- `total_cost`：所有 GPU 总负载。
- `avg_load`：平均负载，`total_cost / num_gpus`。
- `makespan`：最重 GPU 的负载。
- `gap_to_avg`：`makespan - avg_load`。
- `imbalance`：`makespan / avg_load`，越接近 1 越平衡。

---

如果你愿意，我下一条可以把这些变量再整理成一版“老师问你时可以口头直接讲”的简短版本。