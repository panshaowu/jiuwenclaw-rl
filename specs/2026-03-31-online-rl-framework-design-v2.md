# Jiuwen Agent-Core 在线强化学习框架设计文档 (v2.0 - 融合版)

> **版本**: v2.0 (融合版)  
> **日期**: 2026-03-31  
> **状态**: 设计评审  
> **作者**: Sisyphus (AI Agent)  
> **前序版本**: v1.0 (`docs/superpowers/specs/2026-03-31-online-rl-framework-design.md`)  
> **参考草案**: `agent-online-rl-previous-brainstorm/`

---

## 1. 背景与目标

### 1.1 问题陈述

当前 Jiuwen 平台已有一套 RL 方案：
- **离线RL**: agent-core/agentrl - 训练前批量录制轨迹 → 停止Agent → 执行训练 → 重启服务 (需要停机，用户体验差)

为参考在线学习方案，我们调研了第三方开源项目：
- **元学习参考**: MetaClaw - 在线学习，无需GPU (依赖云端Tinker/MinT/Weaver，不兼容verl)
- **强化学习框架参考**: openclaw-rl - 基于slime的异步RL框架 (arXiv:2603.10165)

**核心需求**: 在 agent-core 中构建**在线RL能力**，使智能体在不停机的前提下，一边与用户交互，一边收集轨迹，在合适的时机自动触发 LLM RL 训练。

### 1.2 设计目标

1. **零停机**: Agent 服务持续运行，训练在后台异步执行
2. **最小侵入**: 不颠覆 agent-core 和 jiuwenclaw 的既有架构
3. **生产就绪**: FastAPI 服务、并发安全、流式支持、状态管理
4. **可扩展**: 算法和调度策略未定，预留充分适配空间
5. **可复用**: 复用业界成熟组件（verl、vLLM 等），避免从零实现
6. **泛化性**: agent-core 的在线RL能力可服务于任何基于它的智能体应用

### 1.3 范围界定

**本次设计包含**:
- LLM Gateway (FastAPI 服务: Proxy + SessionRecorder + Reward 外部接口)
- 融合数据格式 (Trajectory + Turn + TokenData + 灵活Reward + 状态机)
- 轨迹存储 (SQLite TrajectoryStore + RolloutPersistence 兼容层)
- 训练调度器 (TriggerPolicy 抽象 + ResourceScheduler 抽象 + BatchUserLoRATrainer)
- RL 后端抽象层 (RLBackend 接口 + 注册机制)
- LoRA 权重管理 (统一版本化存储 + 热加载通知 + 元数据)

**本次设计不包含**:
- 具体 RL 算法实现（GRPO/PPO 等由后端插件提供）
- Reward 计算器的具体算法（预留外部插件接口）
- 多模态数据的具体处理（预留扩展字段）
- 具体调度策略实现（预留策略接口）

### 1.4 相关需求背景

基于 Jiuwen 平台的定位和实际应用场景，在线强化学习框架需要满足以下关键需求：

1. **智能体持续学习能力**：使基于 agent-core 构建的智能体（如 jiuwenclaw）能够从用户交互中持续学习和改进，而不仅仅依赖预训练或离线训练。

2. **多租户支持**：框架需要支持多个用户/智能体实例，每个用户都有独立的学习轨迹和个性化模型（LoRA 权重）。

3. **与现有系统集成**：需要无缝集成到 agent-core 的现有架构中，特别是：
   - 记忆系统：利用已有的上下文管理和记忆压缩机制
   - 权限系统：遵守工具访问控制和用户审批流程
   - 工具系统：支持 MCP 协议和本地工具的调用与记录
   - 通信渠道：适配多平台接入（飞书/钉钉/微信等）的消息格式

4. **实时性和响应性**：在线学习过程不应显著影响智能体的响应时间，训练应在后台异步进行。

5. **可观测性和可调试性**：提供完整的轨迹记录、训练过程监控和结果可视化能力，便于问题定位和效果评估。

6. **故障容错和恢复能力**：在网络中断、服务重启或训练失败等情况下，能够保证数据不丢失并自动恢复。

7. **资源敏感性**：能够根据可用的计算资源（特别是 GPU）动态调整训练频率和批次大小。

8. **合规性和安全性**：确保用户数据的隐私保护和训练过程的安全隔离，符合数据保护 réglementation。

这些需求来源于对 jiuwenclaw 实际应用场景的分析，特别是在客服、个人助理和企业知识管理等需要持续优化和个性化的场景中的应用需求。

---

## 2. verl/slime API 架构分析与设计选型

### 2.1 verl API 架构分析

**源码位置**: `verl/verl/`

#### 2.1.1 数据协议 (DataProto)

verl 使用 `TensorDict` 作为核心数据交换格式，所有训练数据必须转换为张量形式。

```python
# verl/verl/protocol.py
class DataProto:
    """数据交换协议，封装张量批次和元信息"""
    batch: TensorDict              # 张量数据: input_ids, attention_mask, responses, advantages, returns
    non_tensor_batch: dict         # 非张量数据: uid, prompt_str, data_source
    meta_info: dict                # 元信息: 分片信息、填充大小等

    # 核心操作
    concat(data: list[DataProto])  # 合并多个DataProto
    make_iterator(mini_batch_size, epochs)  # mini-batch迭代器
    select(batch_keys, meta_info_keys)      # 选择子集
    to(device)                     # 设备迁移
```

**关键张量字段**:
| 字段 | 形状 | 说明 |
|------|------|------|
| `input_ids` | (batch, seq_len) | 输入token IDs |
| `attention_mask` | (batch, seq_len) | 注意力掩码 |
| `responses` | (batch, response_len) | 模型生成的response tokens |
| `token_level_scores` | (batch, response_len) | token级奖励 |
| `old_log_probs` | (batch, response_len) | 旧策略log概率 |
| `ref_log_prob` | (batch, response_len) | 参考策略log概率 |
| `advantages` | (batch, response_len) | 优势函数 |
| `returns` | (batch, response_len) | 回报值 |
| `response_mask` | (batch, response_len) | response部分掩码 |

#### 2.1.2 训练器接口 (RayPPOTrainer)

```python
# verl/verl/trainer/ppo/ray_trainer.py
class RayPPOTrainer:
    """分布式PPO训练器 (Ray后端)"""
    
    def __init__(
        self,
        config,                                    # OmegaConf配置对象
        tokenizer,                                 # HuggingFace tokenizer
        role_worker_mapping: dict[Role, WorkerType],  # 角色→Worker映射
        resource_pool_manager: ResourcePoolManager,   # GPU资源池管理器
        ray_worker_group_cls = RayWorkerGroup,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        ...
    ):
        # 核心角色:
        #   Role.ActorRollout  → 策略模型 + rollout生成
        #   Role.Critic        → 价值网络 (PPO需要)
        #   Role.Ref           → 参考策略 (KL惩罚)
        #   Role.RewardModel   → 奖励模型

    def fit(self):
        """主训练循环"""
        for step in range(num_steps):
            # 1. 唤醒rollout引擎 (vLLM/SGLang)
            # 2. 生成rollout数据 → DataProto
            # 3. 休眠rollout引擎
            # 4. 计算奖励 (apply_kl_penalty)
            # 5. 计算优势 (compute_advantage: GAE/GRPO/REINFORCE++)
            # 6. 执行PPO更新 (actor/critic)
            # 7. 同步权重到rollout引擎
            # 8. 保存检查点 + 日志
```

**资源管理**: verl 使用 `ResourcePoolManager` 管理 Ray 资源池，通过 `role_worker_mapping` 将不同角色 (Actor/Critic/Ref/Reward) 映射到对应的 Worker 类型。

#### 2.1.3 核心算法函数

```python
# verl/verl/trainer/ppo/core_algos.py
# 优势估计器
AdvantageEstimator.GAE     # GAE (Generalized Advantage Estimation)
AdvantageEstimator.GRPO    # GRPO (Group Relative Policy Optimization)
AdvantageEstimator.PPO     # 标准PPO
AdvantageEstimator.REINFORCE_PLUS_PLUS

# 核心计算函数
compute_gae_advantage_return(token_level_rewards, values, response_mask, gamma, lam)
compute_grpo_outcome_advantage(token_level_rewards, response_mask, index, norm_adv_by_std_in_grpo)
kl_penalty(logprob, ref_logprob, kl_penalty_type)
compute_policy_loss(old_log_prob, log_prob, advantages, response_mask, cliprange)
```

### 2.2 slime API 架构分析

**源码位置**: `openclaw-rl/slime/slime/`

#### 2.2.1 数据格式 (Sample)

slime 使用 Python dataclass，更轻量，适合异步生成场景。

```python
# slime/slime/utils/types.py
@dataclass
class Sample:
    """RL训练样本"""
    prompt: str | list[dict]           # 提示词 (文本或OpenAI消息格式)
    tokens: list[int]                  # 完整token序列 (prompt + response)
    response: str                      # 生成的响应文本
    response_length: int               # 响应token数量
    reward: float | dict               # 奖励值 (标量或字典)
    loss_mask: list[int] | None        # 损失掩码 (1=参与损失, 0=不参与)
    status: Status                     # 状态: PENDING/COMPLETED/TRUNCATED/ABORTED/FAILED
    metadata: dict                     # 自定义元数据
```

**状态枚举**:
```python
class Status(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    TRUNCATED = "truncated"    # 超出最大长度
    ABORTED = "aborted"        # 被中止
    FAILED = "failed"          # 可恢复的失败
```

#### 2.2.2 Rollout 函数接口

```python
# slime 支持自定义rollout函数
def generate_rollout(args, rollout_id, data_source, evaluation=False) 
    -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    """
    args: 完整参数对象
    rollout_id: 当前轮次ID (用于断点恢复)
    data_source: 全局数据源 (获取prompts, 存储部分生成的样本)
    evaluation: 是否为评估模式
    """
```

#### 2.2.3 训练入口

slime 采用 CLI 驱动，通过命令行参数配置所有组件:
```bash
python train.py \
  --actor-num-nodes 1 --actor-num-gpus-per-node 4 \
  --rollout-num-gpus 8 \
  --advantage-estimator grpo \
  --hf-checkpoint /path/to/model \
  --input-key prompt --label-key label
```

### 2.3 agentrl 现有架构分析

**源码位置**: `agent-core/openjiuwen/dev_tools/agentrl/`

#### 2.3.1 数据模型

```python
# agentrl/coordinator/schemas.py
class Rollout(BaseModel):
    """单轮对话rollout"""
    turn_id: Optional[int]
    input_prompt: Dict[str, Any]       # {"message": [...], "tools": [...]}
    output_response: Dict[str, Any]    # OpenAI assistant格式
    llm_config: Dict[str, Any]

class RolloutMessage(BaseModel):
    """完整执行结果 (多轮聚合)"""
    task_id: str
    rollout_info: List[Rollout]        # 所有turn
    reward_list: List[float]           # 每turn奖励
    global_reward: float
    turn_count: int

class RolloutWithReward(BaseModel):
    """Token级训练样本"""
    input_prompt_ids: List[int]        # Token IDs
    output_response_ids: List[int]
    reward: float
    loss_mask: List[int]               # whole-trajectory模式
```

#### 2.3.2 训练协调器

```
MainTrainer (训练循环)
    │
    ├── VerlTrainingExecutor (继承 RayPPOTrainer)
    │       └── train_step() → run_ppo_step()
    │
    └── TrainingCoordinator (Rollout协调)
            ├── ParallelRuntimeExecutor (并行执行)
            │       └── RuntimeExecutor → TrajectoryCollector → TrajectoryCollectionRail
            ├── RolloutEncoder (编码: Rollout → RolloutWithReward)
            └── RLBatchBuilder (批次构建)
```

#### 2.3.3 关键设计亮点

| 组件 | 设计 | 评价 |
|------|------|------|
| **TrajectoryCollectionRail** | AgentRail钩子非侵入式收集 | ✅ 创新，适合agent场景 |
| **VerlTrainingExecutor** | 直接继承 RayPPOTrainer | ✅ 保持原生API兼容 |
| **RewardRegistry** | 奖励函数注册表 | ✅ 灵活可扩展 |
| **Wake/Sleep模式** | 共享GPU资源 | ✅ 高效资源利用 |
| **ProcessorsRegistry** | 分类器/验证器/采样器 | ✅ 灵活的扩展点 |

### 2.4 设计选型逻辑

#### 2.4.1 数据格式选型

| 方案 | 优势 | 劣势 | 选型 |
|------|------|------|------|
| **新建独立数据模型** | 完全自主控制 | 需要额外转换层，维护成本高 | ❌ |
| **直接使用 DataProto** | 与verl零适配成本 | 强依赖verl，难以切换后端 | ⚠️ |
| **Rollout → DataProto 转换层** | 保持agentrl数据模型自主性，同时兼容verl | 需要实现转换逻辑 | ✅ **采用** |

**决策**: 保持 agentrl 的 `Rollout`/`RolloutMessage` 数据模型作为内部表示，在 `RolloutPersistenceAdapter` 中实现到 `DataProto` 的转换。这样既保持代码自主性，又确保与 verl 的无缝对接。

#### 2.4.2 RLBackend 抽象层厚度

| 方案 | 优势 | 劣势 | 选型 |
|------|------|------|------|
| **厚重抽象层** (完整接口定义+注册+适配) | 后端无关，易于切换 | 过度设计，维护成本高 | ❌ |
| **零抽象** (直接调用verl) | 简单直接 | 锁定verl，未来切换成本高 | ⚠️ |
| **轻薄抽象层** (最小接口+verl优先) | 当前简单，未来可扩展 | 需要克制抽象冲动 | ✅ **采用** |

**决策**: 短期内优先接入 verl，RLBackend 抽象层保持最薄:
- 仅定义 `train(data: DataProto) -> TrainingResult` 一个核心方法
- `VerlBackend` 直接包装 `RayPPOTrainer`
- `SlimeBackend` 预留接口，暂不实现

#### 2.4.3 资源管理对齐

| 方案 | 对齐方式 | 复杂度 | 选型 |
|------|---------|--------|------|
| **完全复用 ResourcePoolManager** | 直接使用verl的资源管理 | 高 (需要Ray环境) | ❌ |
| **独立资源调度** | 自建GPU调度逻辑 | 中 (重复造轮子) | ⚠️ |
| **轻量适配层** | 封装verl接口，提供简化API | 低 (最佳平衡) | ✅ **采用** |

**决策**: 在线RL的 `ResourceScheduler` 不直接复用 `ResourcePoolManager`，而是提供简化的资源请求接口，内部委托给 verl 的训练流程处理资源分配。

---

## 3. 整体架构设计 (v2.0 融合版 - 基于 agentrl 基础)

本方案在agent-core现有的离线RL框架（agentrl）基础上进行扩展，以实现在线学习能力。通过复用agentrl的成熟组件（轨迹收集、数据模型、训练后端等），避免代码重复和架构割裂，实现离线和在线RL的统一。

### 3.1 代码结构设计

**重要**: 在线RL框架作为 agentrl 的扩展模块，代码统一放置在 `agent-core/openjiuwen/dev_tools/agentrl/` 下，与离线RL共享同一目录。这确保:
1. 离线/在线RL代码统一管理，避免架构割裂
2. 复用 agentrl 的成熟组件 (TrajectoryCollectionRail, RewardRegistry, VerlTrainingExecutor)
3. 测试代码放置在 `agent-core/tests/unit_tests/dev_tools/agentrl/online/`

```
agent-core/openjiuwen/dev_tools/agentrl/
├── (现有离线RL组件)
│   ├── agent_runtime/          # Agent运行时 + 轨迹收集Rail
│   ├── coordinator/            # 训练协调器
│   ├── rl_trainer/             # verl训练执行器
│   ├── rollout_store/          # 轨迹持久化
│   ├── reward/                 # 奖励注册表
│   ├── proxy/                  # LLM推理代理
│   ├── config/                 # 配置Schema
│   └── monitoring/             # 监控指标
│
└── online/                     # 在线RL扩展模块 (新增)
    ├── __init__.py             # 模块导出
    ├── gateway/                # LLM Gateway (FastAPI服务)
    │   ├── __init__.py
    │   ├── app.py              # FastAPI 应用入口
    │   ├── proxy.py            # HTTP代理 (流式/非流式 + LoRA注入)
    │   └── recorder.py         # SessionRecorder (session生命周期)
    ├── schemas.py              # 融合数据格式 (Trajectory, Turn, TokenData)
    ├── store/                  # 存储层
    │   ├── __init__.py
    │   ├── trajectory_store.py # SQLite TrajectoryStore
    │   └── rollout_adapter.py  # RolloutPersistence → DataProto 转换层
    ├── scheduler/              # 训练调度层
    │   ├── __init__.py
    │   ├── trigger.py          # TriggerPolicy 抽象 + 内置实现
    │   ├── training_scheduler.py  # TrainingScheduler (定时扫描)
    │   └── resource_scheduler.py  # ResourceScheduler (轻量封装)
    ├── backend/                # RL后端 (轻薄抽象)
    │   ├── __init__.py
    │   ├── rl_backend.py       # RLBackend 最小接口
    │   ├── verl_backend.py     # VerlBackend (直接包装RayPPOTrainer)
    │   └── slime_backend.py    # SlimeBackend (预留，暂不实现)
    ├── trainer/                # 训练器
    │   ├── __init__.py
    │   ├── lora_trainer.py     # BatchUserLoRATrainer
    │   └── lora_manager.py     # LoRAManager + 版本管理
    └── config.py               # 在线RL配置Schema
```

### 3.2 架构概览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              agentrl/online/ 在线RL模块 (agentrl扩展)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  LLM Gateway (FastAPI: app.py + proxy.py + recorder.py)             │  │
│  │  - Session生命周期 / 流式SSE收集 / LoRA路由注入 / 并发安全          │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│         │                                              │                    │
│         ▼                                              ▼                    │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  融合数据格式 (schemas.py)                                           │  │
│  │  Trajectory → Turn → TokenData → Reward                            │  │
│  │  ↑ 内部表示，通过 rollout_adapter.py 转换为 verl DataProto          │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│         │                                                                  │
│         ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  轨迹存储层 (store/)                                                 │  │
│  │  TrajectoryStore (SQLite) + RolloutPersistenceAdapter               │  │
│  │  状态机: PENDING → TRAINING → TRAINED / FAILED                      │  │
│  │  ↑ 复用 agentrl 的 RolloutPersistence 接口                          │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│         │                                                                  │
│         ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  训练调度层 (scheduler/)                                             │  │
│  │  TrainingScheduler + TriggerPolicy + ResourceScheduler              │  │
│  │  ↑ ResourceScheduler 轻量封装 verl 资源管理接口                     │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│         │                                                                  │
│         ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  RL后端层 (backend/) - 轻薄抽象                                      │  │
│  │  RLBackend (最小接口: train(data) -> result)                        │  │
│  │  VerlBackend → 直接包装 RayPPOTrainer (优先实现)                    │  │
│  │  SlimeBackend → 预留接口 (暂不实现)                                 │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│         │                                                                  │
│         ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  LoRA权重管理 (trainer/)                                             │  │
│  │  BatchUserLoRATrainer + LoRAManager + 版本化存储 + 热加载通知       │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

与 agentrl 离线RL的关系:
┌─────────────────────────────────────────────────────────────────────────────┐
│  agentrl/ (离线RL)                           │  agentrl/online/ (在线RL)    │
│  ──────────────────────────                  │  ──────────────────────────  │
│  TrajectoryCollectionRail ──────────────────→│  SessionRecorder (复用概念)  │
│  Rollout / RolloutMessage ──────────────────→│  Trajectory (扩展字段)       │
│  RolloutEncoder ────────────────────────────→│  RolloutPersistenceAdapter   │
│  VerlTrainingExecutor ──────────────────────→│  VerlBackend (直接包装)      │
│  RewardRegistry ────────────────────────────→│  RewardCalculator (复用)     │
│  RolloutPersistence ────────────────────────→│  TrajectoryStore (复用接口)  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 组件职责矩阵

| 组件 | 位置 | 职责 | 与 agentrl 关系 |
|------|------|------|-----------------|
| **Gateway (FastAPI)** | `agentrl/online/gateway/app.py` | FastAPI 应用入口 + 路由 | 新增在线服务入口 |
| **Proxy** | `agentrl/online/gateway/proxy.py` | HTTP请求转发 + LoRA路由注入 + 流式SSE | 扩展 agentrl 的 BackendProxy |
| **SessionRecorder** | `agentrl/online/gateway/recorder.py` | Session生命周期 + 流式收集 + 超时清理 | 新增，在线版 TrajectoryCollectionRail |
| **融合数据格式** | `agentrl/online/schemas.py` | Trajectory + Turn + TokenData + Reward + 状态机 | 扩展 agentrl 的 Rollout 数据模型 |
| **TrajectoryStore** | `agentrl/online/store/trajectory_store.py` | SQLite 存储 + 状态管理 + 原子操作 | 复用 agentrl 的 RolloutPersistence 接口 |
| **RolloutPersistenceAdapter** | `agentrl/online/store/rollout_adapter.py` | Trajectory → verl DataProto 转换层 | 核心组件: 连接在线/离线数据格式 |
| **RewardCalculator** | `agentrl/online/scheduler/reward.py` | 外部Reward插件接口 | 复用 agentrl 的 RewardRegistry |
| **TrainingScheduler** | `agentrl/online/scheduler/training_scheduler.py` | 定时扫描 + TriggerPolicy | 新增，在线训练触发 |
| **TriggerPolicy** | `agentrl/online/scheduler/trigger.py` | 可插拔触发策略 | 新增 |
| **ResourceScheduler** | `agentrl/online/scheduler/resource_scheduler.py` | 轻量资源请求接口 | 轻量封装，委托 verl 处理 |
| **RLBackend** | `agentrl/online/backend/rl_backend.py` | 最小训练接口: `train(data) -> result` | 轻薄抽象，verl 优先 |
| **VerlBackend** | `agentrl/online/backend/verl_backend.py` | 直接包装 RayPPOTrainer | 直接复用 VerlTrainingExecutor |
| **BatchUserLoRATrainer** | `agentrl/online/trainer/lora_trainer.py` | 批量顺序LoRA训练 | 新增，在线LoRA训练优化 |
| **LoRAManager** | `agentrl/online/trainer/lora_manager.py` | 统一权重管理 + 版本化 + 热加载 | 新增 |

### 3.4 静态视图

#### 3.4.1 模块划分视图
```
agent-core/openjiuwen/dev_tools/agentrl/online/
├── __init__.py
├── config.py                           # 在线RL配置Schema
├── schemas.py                          # 融合数据格式
├── gateway/                            # LLM Gateway (FastAPI服务)
│   ├── __init__.py
│   ├── app.py                          # FastAPI 应用入口
│   ├── proxy.py                        # HTTP代理 (流式/非流式 + LoRA注入)
│   └── recorder.py                     # SessionRecorder (session生命周期)
├── store/                              # 存储层
│   ├── __init__.py
│   ├── trajectory_store.py             # SQLite TrajectoryStore
│   └── rollout_adapter.py             # Trajectory → verl DataProto 转换层
├── scheduler/                          # 训练调度层
│   ├── __init__.py
│   ├── trigger.py                      # TriggerPolicy 抽象 + 内置实现
│   ├── training_scheduler.py           # TrainingScheduler (定时扫描)
│   └── resource_scheduler.py           # ResourceScheduler (轻量封装)
├── backend/                            # RL后端 (轻薄抽象)
│   ├── __init__.py
│   ├── rl_backend.py                   # RLBackend 最小接口
│   ├── verl_backend.py                 # VerlBackend (直接包装RayPPOTrainer)
│   └── slime_backend.py                # SlimeBackend (预留，暂不实现)
└── trainer/                            # 训练器
    ├── __init__.py
    ├── lora_trainer.py                 # BatchUserLoRATrainer
    └── lora_manager.py                 # LoRAManager + 版本管理
```

#### 3.4.2 依赖关系视图
```
┌─────────────────────────────────────────────────────────────────┐
│                    agentrl/online/gateway/                       │
│  FastAPI App ──→ SessionRecorder ──→ Proxy (LoRA注入)          │
└──────────────────────────────┬──────────────────────────────────┘
                               │ Trajectory
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    agentrl/online/store/                         │
│  TrajectoryStore (SQLite) ──→ RolloutPersistenceAdapter        │
│                                    │                            │
│                                    ▼                            │
│                          Trajectory → DataProto 转换            │
└──────────────────────────────┬──────────────────────────────────┘
                               │ DataProto
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                  agentrl/online/scheduler/                       │
│  TrainingScheduler ──→ TriggerPolicy ──→ ResourceScheduler     │
└──────────────────────────────┬──────────────────────────────────┘
                               │ 训练任务
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                   agentrl/online/backend/                        │
│  RLBackend (最小接口)                                           │
│    └── VerlBackend → RayPPOTrainer (verl)                      │
│    └── SlimeBackend → 预留 (未来)                              │
└──────────────────────────────┬──────────────────────────────────┘
                               │ 训练结果
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                   agentrl/online/trainer/                        │
│  BatchUserLoRATrainer ──→ LoRAManager ──→ 热加载通知            │
└─────────────────────────────────────────────────────────────────┘
```

### 3.5 动态视图

#### 3.5.1 请求处理流程 (Runtime Interaction)
```
用户请求
    ↓
[FastAPI App] (agentrl/online/gateway/app.py)
    → 解析头部 (X-User-ID, X-Session-ID)
    ↓
[SessionRecorder] (agentrl/online/gateway/recorder.py)
    → 记录用户请求 (record_request)
    ↓
[LLM Proxy] (agentrl/online/gateway/proxy.py)
    → 注入活跃 LoRA (_inject_lora)
    → 转发到推理服务 (forward_request/stream_forward)
    ↓
推理服务 (vLLM/SGLang)
    → 返回响应 (流式或非流式)
    ↓
[SessionRecorder]
    → 记录助手响应 (record_response) → 如完成则返回 Trajectory
    ↓
[FastAPI App]
    → 将响应返回给用户
    ↓
[后台处理] (异步)
    → 如有 Trajectory，触发 _handle_trajectory
    ↓
[TrajectoryStore] (agentrl/online/store/trajectory_store.py)
    → 保存 Trajectory (状态: PENDING)
    ↓
[Training Scheduler] (agentrl/online/scheduler/training_scheduler.py)
    → 定时扫描待训练轨迹 (_scan_and_submit)
    ↓
[TriggerPolicy] (agentrl/online/scheduler/trigger.py)
    → 检查是否达到训练阈值
    ↓
[Resource Scheduler] (agentrl/online/scheduler/resource_scheduler.py)
    → 提交批量训练任务
    ↓
[RolloutPersistenceAdapter] (agentrl/online/store/rollout_adapter.py)
    → Trajectory → verl DataProto 转换
    ↓
[VerlBackend] (agentrl/online/backend/verl_backend.py)
    → 调用 RayPPOTrainer.train_step()
    ↓
[LoRA Manager] (agentrl/online/trainer/lora_manager.py)
    → 保存并发布新权重 (publish)
    ↓
[推理服务]
    → 加载新 LoRA 权重 (热更新: /v1/load_lora_adapter)
    ↓
[TrajectoryStore]
    → 更新轨迹状态 (mark_trained/mark_failed)
```

#### 3.5.2 数据流视图 (含格式转换)
```
┌─────────────────┐
│  HTTP 请求/响应  │  OpenAI 兼容格式
└────────┬────────┘
         ▼
┌─────────────────┐
│  SessionRecorder │  → Turn 列表 (用户输入 + 助手输出)
└────────┬────────┘
         ▼
┌─────────────────┐
│   Trajectory     │  agentrl/online/schemas.py
│  (内部表示)      │  - turns: List[Turn]
│                  │  - token_data: TokenData (可选)
│                  │  - reward: RewardSignal
└────────┬────────┘
         ▼
┌─────────────────┐
│ TrajectoryStore  │  SQLite 持久化 (状态机: PENDING→TRAINING→TRAINED/FAILED)
└────────┬────────┘
         ▼
┌─────────────────────────────────────────────────────────────┐
│  RolloutPersistenceAdapter (核心转换层)                      │
│                                                              │
│  Trajectory → agentrl Rollout → RolloutWithReward           │
│       → RolloutEncoder (复用 agentrl)                        │
│       → verl DataProto (TensorDict)                          │
│                                                              │
│  转换映射:                                                    │
│  - prompt_ids      → DataProto.batch["input_ids"]            │
│  - response_ids    → DataProto.batch["responses"]            │
│  - loss_mask       → DataProto.batch["response_mask"]        │
│  - reward          → DataProto.batch["token_level_scores"]   │
│  - user_id         → DataProto.non_tensor_batch["uid"]       │
└────────┬─────────────┘
         ▼ DataProto
┌─────────────────┐
│  VerlBackend     │  RayPPOTrainer.train_step(data)
│  (verl原生)      │  - compute_advantage (GAE/GRPO)
│                  │  - update_actor (PPO更新)
└────────┬────────┘
         ▼
┌─────────────────┐
│  LoRA 权重       │  新训练的用户专属LoRA
└────────┬────────┘
         ▼
┌─────────────────┐
│  热加载通知       │  vLLM /v1/load_lora_adapter
└─────────────────┘
```

#### 3.5.3 控制流视图
```
启动序列:
    应用启动 → 初始化组件 (Gateway, Recorder, Store, Scheduler, Backend)
    ↓
    启动 Training Scheduler 的扫描循环 (后台线程/asyncio task)
    ↓
    进入 FastAPI 请求处理循环

请求处理循环 (同步路径):
    接收 HTTP 请求 → 解析头部
    ↓
    记录请求到 Session (SessionRecorder.record_request)
    ↓
    转发到推理服务 (Proxy.forward_request / stream_forward)
    ↓
    记录响应 (SessionRecorder.record_response)
    ↓
    返回 HTTP 响应给用户

后台处理循环 (异步路径):
    Training Scheduler 定时唤醒 (每N秒)
    ↓
    扫描 TrajectoryStore 查找 PENDING 状态的轨迹
    ↓
    TriggerPolicy 评估是否达到训练阈值 (如: 用户轨迹数 >= min_count)
    ↓
    如有 → 获取待训练轨迹列表，标记为 TRAINING
    ↓
    RolloutPersistenceAdapter 转换 Trajectory → verl DataProto
    ↓
    VerlBackend 调用 RayPPOTrainer 执行训练
    ↓
    训练完成 → 返回权重路径和指标
    ↓
    LoRAManager 保存权重到版本化目录 (v1/, v2/, ...)
    ↓
    LoRAManager 发送热加载通知到推理服务
    ↓
    TrajectoryStore 更新轨迹状态为 TRAINED (或 FAILED)
```

---

## 4. 详细设计 (v2.0 融合版)

### 4.1 融合数据格式

**设计原则**:
- 以之前草案的 `Trajectory` + `Turn` 为基础（完整 session 级别）
- 添加 `TokenData` 支持（录制时 tokenized，避免重复计算）
- Reward 灵活可扩展（`RewardType` 枚举 + `reward_details` Dict）
- 完整状态机（`TrajectoryStatus`: PENDING → TRAINING → TRAINED/FAILED）

```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class TrajectoryStatus(str, Enum):
    """轨迹生命周期状态"""
    PENDING = "pending"      # 待训练
    TRAINING = "training"    # 训练中
    TRAINED = "trained"      # 已训练
    FAILED = "failed"        # 训练失败


class RewardType(str, Enum):
    """Reward 信号来源类型"""
    PRM = "prm"              # Process Reward Model
    ENV = "env"              # 环境奖励 (如任务完成)
    HUMAN = "human"          # 人工标注
    CUSTOM = "custom"        # 自定义


@dataclass
class Turn:
    """单轮对话"""
    role: str                          # "user" | "assistant" | "tool"
    content: str
    timestamp: datetime
    token_count: int = 0
    tool_calls: Optional[List[Dict[str, Any]]] = None


@dataclass
class TokenData:
    """Token 级训练数据 (录制时 tokenized)"""
    prompt_ids: List[int]
    response_ids: List[int]
    logprobs: List[float] = field(default_factory=list)
    loss_mask: List[int] = field(default_factory=list)
    # 多模态扩展 (预留)
    multimodal_tokens: Optional[Dict[str, Any]] = None


@dataclass
class LoRAVersion:
    """LoRA 版本元数据"""
    user_id: str
    version: str                       # "v1", "v2", ...
    path: str
    created_at: datetime
    trajectory_count: int
    reward_avg: float
    base_model: str
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    """
    融合轨迹格式 - 完整 session 级别 + TokenData + 灵活Reward + 状态机
    
    设计来源:
    - Trajectory + Turn: 之前草案 (完整 session 生命周期)
    - TokenData: 当前设计 (录制时 tokenized，避免重复计算)
    - RewardType + reward_details: 当前设计 (灵活扩展)
    - TrajectoryStatus: 之前草案 (完整状态机)
    """
    
    # 基础标识
    trajectory_id: str
    user_id: str
    session_id: str
    
    # 完整对话 (多轮)
    turns: List[Turn]
    created_at: datetime
    
    # Token 级数据 (录制时 tokenized，可选)
    token_data: Optional[TokenData] = None
    
    # 奖励信号 (灵活扩展)
    reward: Optional[float] = None     # [-1, 1] 或任意范围
    reward_type: RewardType = RewardType.CUSTOM
    reward_details: Dict[str, Any] = field(default_factory=dict)
    
    # 状态机
    status: TrajectoryStatus = TrajectoryStatus.PENDING
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**设计辨析**:

| 选择 | 方案 | 理由 |
|------|------|------|
| 数据粒度 | Session 级别 (List[Turn]) | 完整对话上下文，更符合"轨迹"概念 |
| Tokenize 时机 | 录制时 (可选) | 避免重复计算，但保留灵活性 |
| Reward 类型 | 枚举 + 灵活 details | 算法未定，预留扩展空间 |
| 状态管理 | 完整状态机 | 生产环境必需 |
| 多模态支持 | 预留 multimodal_tokens | 优先级低，先支持文本 |

---

### 4.2 LLM Gateway (FastAPI 服务)

**职责**: 生产就绪的 LLM 代理服务

```python
# app.py - FastAPI 应用入口
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI

from openjiuwen.core.llm.gateway.proxy import LLMProxy
from openjiuwen.core.llm.gateway.recorder import SessionRecorder
from openjiuwen.core.llm.gateway.lora_manager import LoRAManager

_store: Optional["TrajectoryStore"] = None
_lora_repo: Optional["LoRAManager"] = None
_recorder: Optional[SessionRecorder] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global _store, _lora_repo, _recorder
    # 初始化组件
    yield
    # 清理资源


def create_app() -> FastAPI:
    app = FastAPI(title="Jiuwen Online RL Gateway", lifespan=lifespan)
    
    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        # 1. 解析 headers
        user_id = request.headers.get("X-User-ID", "anonymous")
        session_id = request.headers.get("X-Session-ID", str(uuid.uuid4()))
        
        body = await request.json()
        
        # 2. 注入 LoRA
        latest_lora = await _lora_repo.get_active_lora(user_id)
        if latest_lora:
            body.setdefault("extra_body", {})["lora_name"] = latest_lora
        
        # 3. 录制请求
        _recorder.record_request(session_id, user_id, body.get("messages", []))
        
        # 4. 转发 (支持流式/非流式)
        is_stream = body.get("stream", False)
        if is_stream:
            return await _stream_forward(body, session_id)
        else:
            return await _forward(body, session_id)
    
    return app
```

**Proxy 流式支持**:

```python
# proxy.py - 流式转发
async def _stream_forward(body: dict, session_id: str) -> StreamingResponse:
    """流式转发：透传 SSE 给 Agent，同时收集完整响应用于录制。"""
    collected_content = []
    finish_reason = None

    async def generate():
        nonlocal finish_reason
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                f"{_inference_url}/v1/chat/completions",
                json=body,
            ) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            chunk = json.loads(line[6:])
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            if delta.get("content"):
                                collected_content.append(delta["content"])
                            fr = chunk.get("choices", [{}])[0].get("finish_reason")
                            if fr:
                                finish_reason = fr
                        except Exception:
                            pass
                    yield line + "\n"

        # 流结束后录制
        if finish_reason == "stop":
            fake_response = {
                "choices": [{
                    "message": {"content": "".join(collected_content), "role": "assistant"},
                    "finish_reason": "stop",
                }]
            }
            trajectory = _recorder.record_response(session_id, fake_response)
            if trajectory:
                asyncio.create_task(_handle_trajectory(trajectory))

    return StreamingResponse(generate(), media_type="text/event-stream")
```

---

### 4.3 SessionRecorder (完整 Session 生命周期)

```python
# recorder.py
import threading
import uuid
from datetime import datetime
from typing import Optional

from openjiuwen.core.llm.gateway.schemas import Trajectory, Turn

SESSION_TTL = 3600  # session 超时（秒）


class SessionRecorder:
    """管理所有 session 的轨迹录制生命周期（内存存储，生产可替换为 Redis）。"""

    def __init__(self):
        self._sessions: dict[str, dict] = {}  # session_id → session state
        self._lock = threading.Lock()

    def record_request(self, session_id: str, user_id: str, messages: list) -> None:
        """录制用户请求，更新 session 上下文。"""
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = {
                    "user_id": user_id,
                    "session_id": session_id,
                    "turns": [],
                    "created_at": datetime.now(),
                    "last_active": datetime.now(),
                }
            session = self._sessions[session_id]
            session["last_active"] = datetime.now()

            # 提取最后一条 user 消息（避免重复录制历史消息）
            user_msgs = [m for m in messages if m.get("role") == "user"]
            if user_msgs:
                last_user = user_msgs[-1]
                session["turns"].append(Turn(
                    role="user",
                    content=last_user.get("content", ""),
                    timestamp=datetime.now(),
                ))

    def record_response(self, session_id: str, response: dict) -> Optional[Trajectory]:
        """
        录制 Assistant 响应。
        若 finish_reason == 'stop'，返回完整 Trajectory 并清理 session。
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return None

            choices = response.get("choices", [])
            if not choices:
                return None

            choice = choices[0]
            message = choice.get("message", {})
            finish_reason = choice.get("finish_reason", "")

            session["turns"].append(Turn(
                role="assistant",
                content=message.get("content") or "",
                timestamp=datetime.now(),
                token_count=response.get("usage", {}).get("completion_tokens", 0),
            ))
            session["last_active"] = datetime.now()

            if finish_reason == "stop":
                return self._finalize_session(session_id, session)
            return None

    def on_session_timeout(self, session_id: str) -> Optional[Trajectory]:
        """session 超时，强制结束录制。"""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return None
            return self._finalize_session(session_id, session)

    def _finalize_session(self, session_id: str, session: dict) -> Optional[Trajectory]:
        """构建 Trajectory 并从内存中移除 session（调用方持有 lock）。"""
        turns = session.get("turns", [])
        if not turns:
            self._sessions.pop(session_id, None)
            return None

        trajectory = Trajectory(
            trajectory_id=str(uuid.uuid4()),
            user_id=session["user_id"],
            session_id=session_id,
            turns=turns,
            created_at=session["created_at"],
        )
        self._sessions.pop(session_id, None)
        return trajectory
```

---

### 4.4 Reward 计算 (外部插件接口)

**设计原则**: 
- 外部化，不绑定具体算法
- 支持 LLM-as-Judge、环境奖励、人工标注等多种模式
- 容错机制（之前草案的 JSON 解析容错）

```python
# reward.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from openjiuwen.core.llm.gateway.schemas import Trajectory, RewardType


class RewardCalculator(ABC):
    """Reward 计算器抽象接口"""
    
    @abstractmethod
    async def compute(self, trajectory: Trajectory) -> Trajectory:
        """计算 reward 并填入 trajectory"""
        raise NotImplementedError
    
    @abstractmethod
    def get_reward_type(self) -> RewardType:
        """返回 reward 类型"""
        raise NotImplementedError


class ExternalRewardCalculator(RewardCalculator):
    """
    外部 Reward 计算器 - 调用外部服务/插件
    
    支持:
    - LLM-as-Judge (之前草案的 RewardComputor 可作为默认实现)
    - 环境 Reward
    - 人工标注
    - 自定义 Reward 服务
    """
    
    def __init__(self, endpoint: str, reward_type: RewardType = RewardType.CUSTOM):
        self.endpoint = endpoint
        self._reward_type = reward_type
    
    async def compute(self, trajectory: Trajectory) -> Trajectory:
        """调用外部 Reward 服务"""
        try:
            # 调用外部服务
            result = await self._call_external_service(trajectory)
            trajectory.reward = result.get("reward", 0.0)
            trajectory.reward_details = result.get("details", {})
            trajectory.reward_type = self._reward_type
        except Exception as e:
            # 容错：计算失败时填入中性 reward
            trajectory.reward = 0.0
            trajectory.reward_details = {"error": str(e)}
        return trajectory
    
    async def _call_external_service(self, trajectory: Trajectory) -> Dict[str, Any]:
        """调用外部 Reward 服务（具体实现由插件提供）"""
        raise NotImplementedError
    
    def get_reward_type(self) -> RewardType:
        return self._reward_type
```

---

### 4.5 轨迹存储 (SQLite + 状态机)

```python
# store/trajectory_store.py
import json
import threading
from datetime import datetime
from pathlib import Path

from openjiuwen.core.llm.gateway.schemas import (
    Trajectory, TrajectoryStatus, Turn
)


class TrajectoryStore:
    """SQLite 轨迹存储，支持完整状态机"""
    
    def __init__(self, db_path: str = "trajectories.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()
    
    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trajectories (
                    trajectory_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    turns_json TEXT NOT NULL,
                    token_data_json TEXT,
                    created_at TEXT NOT NULL,
                    reward REAL,
                    reward_type TEXT DEFAULT 'custom',
                    reward_details_json TEXT DEFAULT '{}',
                    status TEXT NOT NULL DEFAULT 'pending',
                    metadata_json TEXT DEFAULT '{}'
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_status ON trajectories (user_id, status)")
    
    def save(self, trajectory: Trajectory) -> None:
        """保存轨迹"""
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO trajectories
                (trajectory_id, user_id, session_id, turns_json, token_data_json,
                 created_at, reward, reward_type, reward_details_json, status, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trajectory.trajectory_id,
                trajectory.user_id,
                trajectory.session_id,
                json.dumps([{"role": t.role, "content": t.content,
                             "timestamp": t.timestamp.isoformat(), "token_count": t.token_count}
                            for t in trajectory.turns]),
                json.dumps(trajectory.token_data.__dict__) if trajectory.token_data else None,
                trajectory.created_at.isoformat(),
                trajectory.reward,
                trajectory.reward_type.value,
                json.dumps(trajectory.reward_details),
                trajectory.status.value,
                json.dumps(trajectory.metadata),
            ))
    
    def get_pending_count(self, user_id: str) -> int:
        """获取用户待训练轨迹数量"""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM trajectories WHERE user_id=? AND status=?",
                (user_id, TrajectoryStatus.PENDING.value)
            ).fetchone()
            return row[0]
    
    def get_users_above_threshold(self, threshold: int) -> list[str]:
        """获取所有达到阈值的用户"""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT user_id, COUNT(*) as cnt
                FROM trajectories
                WHERE status=?
                GROUP BY user_id
                HAVING cnt >= ?
            """, (TrajectoryStatus.PENDING.value, threshold)).fetchall()
            return [row["user_id"] for row in rows]
    
    def fetch_and_mark_training(self, user_id: str, limit: int) -> list[Trajectory]:
        """原子操作：获取 PENDING 轨迹并标记为 TRAINING"""
        with self._lock:
            with self._conn() as conn:
                rows = conn.execute("""
                    SELECT * FROM trajectories
                    WHERE user_id=? AND status=?
                    LIMIT ?
                """, (user_id, TrajectoryStatus.PENDING.value, limit)).fetchall()
                
                if not rows:
                    return []
                
                ids = [r["trajectory_id"] for r in rows]
                placeholders = ",".join("?" * len(ids))
                conn.execute(
                    f"UPDATE trajectories SET status=? WHERE trajectory_id IN ({placeholders})",
                    [TrajectoryStatus.TRAINING.value] + ids,
                )
                trajectories = [self._row_to_trajectory(r) for r in rows]
                for traj in trajectories:
                    traj.status = TrajectoryStatus.TRAINING
                return trajectories
    
    def mark_trained(self, trajectory_ids: list[str]) -> None:
        self._update_status(trajectory_ids, TrajectoryStatus.TRAINED)
    
    def mark_failed(self, trajectory_ids: list[str]) -> None:
        self._update_status(trajectory_ids, TrajectoryStatus.FAILED)
    
    def _update_status(self, trajectory_ids: list[str], status: TrajectoryStatus) -> None:
        if not trajectory_ids:
            return
        placeholders = ",".join("?" * len(trajectory_ids))
        with self._conn() as conn:
            conn.execute(
                f"UPDATE trajectories SET status=? WHERE trajectory_id IN ({placeholders})",
                [status.value] + trajectory_ids,
            )
```

---

### 4.6 训练调度 (融合版)

```python
# scheduler.py
import asyncio
import logging
from typing import TYPE_CHECKING, List, Optional

from openjiuwen.core.llm.gateway.trigger import TriggerPolicy
from openjiuwen.core.llm.gateway.resource_scheduler import ResourceScheduler

if TYPE_CHECKING:
    from openjiuwen.core.llm.gateway.store.trajectory_store import TrajectoryStore

logger = logging.getLogger(__name__)


class TrainingScheduler:
    """训练调度器 - 融合 TriggerPolicy + ResourceScheduler"""
    
    def __init__(
        self,
        trajectory_store: "TrajectoryStore",
        trigger_policy: TriggerPolicy,
        resource_scheduler: ResourceScheduler,
        scan_interval: float = 600.0,  # 10 分钟
    ):
        self.trajectory_store = trajectory_store
        self.trigger_policy = trigger_policy
        self.resource_scheduler = resource_scheduler
        self.scan_interval = scan_interval
        self._running = False
    
    async def start(self):
        self._running = True
        asyncio.create_task(self._scan_loop())
    
    async def stop(self):
        self._running = False
    
    async def _scan_loop(self):
        while self._running:
            await self._scan_and_submit()
            await asyncio.sleep(self.scan_interval)
    
    async def _scan_and_submit(self):
        """扫描并批量提交训练任务"""
        threshold = self.trigger_policy.get_threshold()
        eligible_users = await self.trajectory_store.get_users_above_threshold(threshold)
        
        if not eligible_users:
            return
        
        # 批量提交
        user_jobs = []
        for user_id in eligible_users:
            trajectories = self.trajectory_store.fetch_and_mark_training(
                user_id, limit=1000
            )
            if trajectories:
                user_jobs.append({
                    "user_id": user_id,
                    "trajectory_ids": [t.trajectory_id for t in trajectories],
                })
        
        if user_jobs:
            job_id = self.resource_scheduler.submit_batch_training_job(user_jobs)
            logger.info("Submitted batch training job: %s for %d users", job_id, len(user_jobs))
```

```python
# resource_scheduler.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass
class UserTrainingJob:
    user_id: str
    trajectory_ids: list[str]


@dataclass
class JobStatus:
    job_id: str
    status: str  # "running" | "completed" | "failed"
    result: dict = None


class ResourceScheduler(ABC):
    """资源调度器抽象接口"""
    
    @abstractmethod
    def submit_batch_training_job(self, user_jobs: List[UserTrainingJob]) -> str:
        """提交批量训练任务，返回 job_id"""
        raise NotImplementedError
    
    @abstractmethod
    def get_job_status(self, job_id: str) -> JobStatus:
        raise NotImplementedError
    
    @abstractmethod
    def cancel_job(self, job_id: str) -> None:
        raise NotImplementedError


class LocalProcessScheduler(ResourceScheduler):
    """本地进程实现（开发调试用）"""
    pass


class K8sJobScheduler(ResourceScheduler):
    """K8s Job 实现"""
    pass


class RayJobScheduler(ResourceScheduler):
    """Ray Job 实现（与 verl 天然集成）"""
    pass
```

---

### 4.7 BatchUserLoRATrainer (批量顺序训练)

```python
# trainer.py
from openjiuwen.core.llm.gateway.rl_backend import RLBackend
from openjiuwen.core.llm.gateway.lora_manager import LoRAManager


class BatchUserLoRATrainer:
    """
    基础模型只加载一次，全程冻结。
    LoRA 适配器在用户间顺序重置复用。
    """
    
    def __init__(self, rl_backend: RLBackend, lora_manager: LoRAManager):
        self.rl_backend = rl_backend
        self.lora_manager = lora_manager
    
    def run(self, user_batch: List[dict]):
        for job in user_batch:
            try:
                self._train_one_user(job)
            except Exception as e:
                # 单用户失败不影响其他用户
                logger.error(f"Training failed for user {job['user_id']}: {e}")
    
    def _train_one_user(self, job: dict):
        # 1. 加载轨迹数据
        # 2. 执行训练
        result = await self.rl_backend.train(trajectories)
        # 3. 发布权重
        await self.lora_manager.publish(job["user_id"], result.weights_path)
        # 4. 标记已训练
        self.trajectory_store.mark_trained(job["trajectory_ids"])
```

---

### 4.8 LoRA 权重管理 (融合版)

```python
# lora_manager.py
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import httpx


@dataclass
class LoRAVersion:
    """LoRA 版本元数据"""
    user_id: str
    version: str
    path: str
    created_at: datetime
    trajectory_count: int
    reward_avg: float
    base_model: str
    metrics: Dict[str, Any] = field(default_factory=dict)


class LoRAStorageAdapter(ABC):
    """LoRA 存储适配器抽象"""
    
    @abstractmethod
    async def save(self, user_id: str, version: int, weights_path: str, metadata: dict) -> None:
        pass
    
    @abstractmethod
    async def get_latest_version(self, user_id: str) -> Optional[int]:
        pass
    
    @abstractmethod
    async def set_latest(self, user_id: str, version: int) -> None:
        pass
    
    @abstractmethod
    async def get_version_metadata(self, user_id: str, version: int) -> Optional[LoRAVersion]:
        pass


class LocalStorageAdapter(LoRAStorageAdapter):
    """本地文件系统存储"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
    
    async def save(self, user_id, version, weights_path, metadata):
        dest = self.base_path / user_id / f"v{version}"
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copytree(weights_path, dest, dirs_exist_ok=True)
        # 保存元数据
        import json
        (dest / "metadata.json").write_text(json.dumps(metadata, default=str))
    
    async def get_latest_version(self, user_id):
        user_dir = self.base_path / user_id
        if not user_dir.exists():
            return None
        versions = [int(d.name[1:]) for d in user_dir.iterdir() if d.is_dir() and d.name.startswith("v")]
        return max(versions) if versions else None
    
    async def set_latest(self, user_id, version):
        latest_link = self.base_path / user_id / "latest"
        target = self.base_path / user_id / f"v{version}"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(target)


class InferenceNotifier:
    """推理服务通知器"""
    
    def __init__(self, inference_url: str):
        self.inference_url = inference_url
        self.client = httpx.AsyncClient()
    
    async def notify_update(self, user_id: str, lora_name: str, weights_path: str):
        """通知推理服务加载新 LoRA"""
        await self.client.post(
            f"{self.inference_url}/v1/load_lora_adapter",
            json={"lora_name": lora_name, "lora_path": weights_path},
        )


class LoRAManager:
    """LoRA 权重管理器"""
    
    def __init__(self, storage: LoRAStorageAdapter, notifier: InferenceNotifier):
        self.storage = storage
        self.notifier = notifier
        self._active_loras: Dict[str, str] = {}
    
    async def get_active_lora(self, user_id: str) -> Optional[str]:
        return self._active_loras.get(user_id)
    
    async def publish(self, user_id: str, weights_path: str, metadata: dict = None) -> None:
        version = await self._next_version(user_id)
        await self.storage.save(user_id, version, weights_path, metadata or {})
        await self.storage.set_latest(user_id, version)
        
        lora_name = f"{user_id}_v{version}"
        await self.notifier.notify_update(user_id, lora_name, weights_path)
        self._active_loras[user_id] = lora_name
```

---

## 5. 目录结构 (v2.0)

```
agent-core/openjiuwen/core/llm/gateway/
├── __init__.py                      # 模块入口
├── app.py                           # FastAPI 应用入口 + lifespan
├── proxy.py                         # HTTP 代理 (流式/非流式 + LoRA 注入)
├── recorder.py                      # SessionRecorder (session 生命周期)
├── schemas.py                       # 融合数据格式 (Trajectory, Turn, TokenData, etc.)
├── reward.py                        # RewardCalculator 外部接口
├── trigger.py                       # TriggerPolicy 抽象 + 内置实现
├── scheduler.py                     # TrainingScheduler (定时扫描)
├── resource_scheduler.py            # ResourceScheduler 抽象 + 内置实现
├── rl_backend.py                    # RLBackend 抽象 + 注册表
├── trainer.py                       # BatchUserLoRATrainer
├── lora_manager.py                  # LoRAManager + LoRAStorageAdapter + InferenceNotifier
├── store/
│   ├── __init__.py
│   ├── trajectory_store.py          # SQLite TrajectoryStore
│   └── rollout_adapter.py           # RolloutPersistence 兼容层
├── backends/
│   ├── __init__.py
│   └── verl_backend.py              # VerlBackend (占位)
└── config.py                        # Gateway 配置 Schema
```

---

## 6. 设计决策汇总 (v2.0)

| 决策点 | 选择 | 来源 | 理由 |
|--------|------|------|------|
| 架构定位 | agent-core 内部模块 | 当前设计 | 泛化性，零额外部署 |
| Gateway 框架 | FastAPI | 之前草案 | 生产就绪，流式支持 |
| Session 管理 | SessionRecorder (内存+Lock) | 之前草案 | 完整生命周期，并发安全 |
| 数据粒度 | Session 级别 (List[Turn]) | 之前草案 | 完整对话上下文 |
| Tokenize 时机 | 录制时 (可选) | 当前设计 | 避免重复计算 |
| Reward 计算 | 外部插件接口 | 当前设计 | 算法未定，灵活适配 |
| Reward 容错 | 降级为中性 reward | 之前草案 | 生产健壮性 |
| 状态管理 | 完整状态机 (4状态) | 之前草案 | 生产必需 |
| 训练触发 | TriggerPolicy 抽象 | 当前设计 | 策略灵活 |
| 资源调度 | ResourceScheduler 抽象 | 之前草案 | 支持多后端 |
| 批量训练 | BatchUserLoRATrainer | 之前草案 | 高效共享基础模型 |
| RL 后端 | RLBackend 注册表 | 当前设计 | 多框架支持 |
| LoRA 存储 | 抽象适配器 | 当前设计 | 云存储适配 |
| LoRA 元数据 | LoRAVersion 详细 | 之前草案 | 版本管理 |

---

## 7. 与 jiuwenclaw 的集成

*(同 v1.0)*

---

## 8. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| Tokenizer 与推理服务不一致 | 训练数据偏差 | 使用相同模型版本的 tokenizer |
| 训练时 GPU 资源不足 | 训练失败 | ResourceScheduler 检测资源可用性 |
| LoRA 热加载失败 | 用户使用旧权重 | 通知器重试 + 回滚机制 |
| 轨迹数据量过大 | 存储压力 | 定期归档 + 压缩 |
| Session 内存泄漏 | 服务崩溃 | SESSION_TTL 超时清理 + 监控 |
| Reward 计算失败 | 中性 reward 降级 | 容错机制，不阻塞主链路 |

---

## 9. 后续工作

1. **详细实现计划**: 基于此设计文档，编写分步实现计划
2. **RL 算法对接**: 与算法团队确认 verl 后端的具体接口
3. **Reward 插件**: 实现 LLM-as-Judge 默认插件
4. **性能测试**: 验证 Gateway 代理的延迟影响
5. **多模态扩展**: 后续支持图片、音频等模态

---

> **文档结束**
> 
> 本文档为 Jiuwen agent-core 在线强化学习框架的融合设计（v2.0）。
> 融合了 `agent-online-rl-previous-brainstorm/` 的生产就绪设计和当前设计的可扩展架构。
