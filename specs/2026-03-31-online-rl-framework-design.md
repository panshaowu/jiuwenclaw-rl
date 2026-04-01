# Jiuwen Agent-Core 在线强化学习框架设计文档

> **版本**: v1.0 (已废弃，见 v2.0)  
> **日期**: 2026-03-31  
> **状态**: 设计评审 → 已被 v2.0 融合版替代  
> **作者**: Sisyphus (AI Agent)  
> **替代文档**: `docs/superpowers/specs/2026-03-31-online-rl-framework-design-v2.md`

---

## 1. 背景与目标

### 1.1 问题陈述

当前 Jiuwen 平台已有两套 RL 方案：

| 方案 | 项目 | 定位 | 局限 |
|------|------|------|------|
| **离线RL** | agent-core/agentrl | 训练前批量录制轨迹 → 停止Agent → 执行训练 → 重启服务 | 需要停机，用户体验差 |
| **元学习** | MetaClaw | 在线学习，无需GPU | 依赖云端Tinker/MinT/Weaver，不兼容verl |

**核心需求**: 在 agent-core 中构建**在线RL能力**，使智能体在不停机的前提下，一边与用户交互，一边收集轨迹，在合适的时机自动触发 LLM RL 训练。

### 1.2 设计目标

1. **零停机**: Agent 服务持续运行，训练在后台异步执行
2. **最小侵入**: 不颠覆 agent-core 和 jiuwenclaw 的既有架构
3. **可扩展**: 算法仍在设计中，预留适配空间
4. **可复用**: 复用业界成熟组件（verl、vLLM 等），避免从零实现
5. **泛化性**: agent-core 的在线RL能力可服务于任何基于它的智能体应用

### 1.3 范围界定

**本次设计包含**:
- LLM Gateway（Proxy + Recorder）
- 统一轨迹格式（UnifiedTrajectory）
- 轨迹存储（复用 RolloutPersistence 接口）
- 训练调度器（TriggerPolicy + Scheduler）
- RL 后端抽象层（RLBackend 接口 + 注册机制）
- LoRA 权重管理（统一版本化存储 + 热加载通知）

**本次设计不包含**:
- 具体 RL 算法实现（GRPO/PPO 等由后端插件提供）
- Reward 计算器的具体算法（预留扩展接口）
- 多模态数据的具体处理（预留扩展字段）

---

## 2. 关联项目架构分析

### 2.1 agent-core (openjiuwen)

**定位**: Jiuwen 平台的 Agent 核心框架，提供高性能 Agent 运行时。

**架构**:
```
agent-core/openjiuwen/
├── core/                    # 核心引擎 (17个子模块)
│   ├── application/         # SDK接口层 (ReActAgent, WorkflowAgent)
│   ├── runner/              # 执行引擎 (异步并行图执行)
│   ├── workflow/            # 工作流编排 (组件并发、动态跳转)
│   ├── foundation/          # 基础层 (LLM/Tool/Storage)
│   │   └── llm/
│   │       └── model_clients/  # LLM客户端 (OpenAI/DashScope等)
│   └── single_agent/        # 单Agent基类 (ReActAgent)
├── dev_tools/
│   └── agentrl/             # 离线RL训练框架 (34文件, 5442行)
│       ├── optimizer/       # 用户入口 (RLOptimizer)
│       ├── rl_trainer/      # PPO训练 (VerlTrainingExecutor)
│       ├── coordinator/     # 训练协调 (TrainingCoordinator)
│       ├── agent_runtime/   # Agent运行时 (TrajectoryCollector)
│       └── rollout_store/   # 轨迹持久化 (FileRolloutStore)
└── extensions/              # 扩展模块
```

**关键发现**:
- LLM 调用是**直连模式**: Agent → BaseModelClient → LLM API
- **没有** LLM 代理层或拦截层
- 已有 `RolloutPersistence` 抽象接口和 `FileRolloutStore` 实现
- 已有 `TrajectoryCollectionRail` 通过 RAIL 钩子收集轨迹

### 2.2 jiuwenclaw

**定位**: 基于 agent-core 构建的智能 AI 助手应用。

**架构关系**:
```
jiuwenclaw (应用层)
    └── 继承 openjiuwen.core.single_agent.ReActAgent
    └── 依赖 openjiuwen.core.foundation (Model, Session, ToolInfo)
    └── pyproject.toml: "openjiuwen==0.1.7"
```

**Gateway 职责**: 消息路由枢纽（Channel ↔ AgentServer），**不直接与 LLM 交互**。

**关键发现**:
- jiuwenclaw 是 agent-core 的**上层应用**
- LLM 调用发生在 AgentServer 内部（react_agent.py）
- 在线RL 应放在 agent-core 中，保持泛化性

### 2.3 openclaw-rl

**定位**: 全异步 RL 框架，训练个性化 AI Agent。论文: arXiv:2603.10165

**架构**:
```
openclaw-rl/
├── slime/                   # RL训练核心框架 (THUDM)
├── terminal-rl/             # 终端智能体RL
├── swe-rl/                  # SWE智能体RL
├── toolcall-rl/             # 工具调用RL
├── gui-rl/                  # GUI智能体RL
├── openclaw-rl/             # Binary RL实现
├── openclaw-opd/            # On-Policy Distillation
└── openclaw-combine/        # 组合方法
```

**核心特性**:
- 全异步4组件架构: Agent服务 → Rollout收集 → PRM/Judge评估 → 策略训练
- 三种优化方法: Binary RL, OPD (On-Policy Distillation), Combined
- 训练后端: slime (THUDM)
- 需要 8x GPU 本地部署

**数据格式**:
```python
@dataclass
class ConversationSample:
    session_id: str
    turn_num: int
    prompt_tokens: list[int]           # 已tokenize
    response_tokens: list[int]
    response_logprobs: list[float]     # 每步log概率
    loss_mask: list[int]
    reward: float                      # {-1, 0, 1} 离散
    prompt_text: str = ""
    response_text: str = ""
    teacher_logprobs: Optional[list[float]] = None  # OPD
    skill_generation: int = 0          # 技能版本号
```

### 2.4 MetaClaw

**定位**: 在线元学习和演化框架，无需GPU。论文: arXiv:2603.17187

**架构**:
```
MetaClaw/
├── metaclaw/
│   ├── api_server.py          # FastAPI代理服务器 (LLM Proxy)
│   ├── skill_manager.py       # 技能管理
│   ├── skill_evolver.py       # 技能演化
│   ├── trainer.py             # RL训练器
│   ├── scheduler.py           # MadMax智能调度器
│   ├── prm_scorer.py          # PRM评分
│   ├── rollout.py             # Rollout收集
│   └── memory/                # 长期记忆系统
└── openclaw-metaclaw-memory/  # 记忆Sidecar服务
```

**核心特性**:
- 三种模式: `skills_only` / `rl` / `madmax` (默认)
- MadMax 智能调度: 睡眠时段/键盘空闲/日历事件触发训练
- 技能演化: 从失败经验自动提取新技能
- 长期记忆: 跨会话记忆 (事实/偏好/项目状态)
- 训练后端: Tinker/MinT/Weaver (云端，无需GPU)

**训练流程**:
```
Rollout收集 → PRM评分 → GRPO优势计算 → Tinker LoRA训练 → 权重热更新 → 技能演化
```

**数据格式**: 与 openclaw-rl 相同 (ConversationSample)

### 2.5 四个项目的对比

| 维度 | agent-core (offline) | openclaw-rl | MetaClaw | 在线RL (目标) |
|------|---------------------|-------------|----------|---------------|
| **训练时机** | 停机训练 | 批量离线 | 在线+智能调度 | 在线+智能调度 |
| **训练后端** | verl | slime | Tinker/MinT/Weaver | verl (优先) + 可扩展 |
| **GPU需求** | 可选 | 8x GPU | 无 (云端) | 外部GPU集群 |
| **轨迹收集** | RAIL钩子 | 直接记录 | API Proxy | LLM Gateway |
| **数据存储** | 本地JSONL | 本地 | 本地 | 可插拔存储 |
| **技能系统** | 无 | 手动 | 自动演化 | 预留接口 |
| **记忆系统** | 无 | 无 | 内置 | 预留接口 |
| **调度器** | 无 | 无 | MadMax | 可自定义触发 |

---

## 3. 整体架构设计

### 3.1 架构概览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         agent-core 在线RL 模块                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                      LLM Gateway                                    │  │
│  │                                                                     │  │
│  │  ┌──────────────────────┐    ┌──────────────────────────────────┐   │  │
│  │  │       Proxy         │    │         Recorder                 │   │  │
│  │  │  - HTTP请求转发     │    │  - 轨迹录制                      │   │  │
│  │  │  - LoRA路由注入     │    │  - Token化 (录制时)              │   │  │
│  │  │  - 统一轨迹格式     │    │  - RolloutPersistence接口       │   │  │
│  │  └──────────────────────┘    └──────────────────────────────────┘   │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                     │                                       │
│                                     ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                    轨迹存储层                                        │  │
│  │     (RolloutPersistence 接口 + UnifiedTrajectory 格式)              │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                     │                                       │
│                                     ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                    训练调度器                                        │  │
│  │     - 定时扫描                                                       │  │
│  │     - 阈值触发                                                      │  │
│  │     - 扩展点: TriggerPolicy (自定义触发条件)                        │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                     │                                       │
│                                     ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                    RL 后端抽象层                                     │  │
│  │                                                                     │  │
│  │  ┌─────────────────────────────────────────────────────────────┐   │  │
│  │  │              RLBackend (抽象接口)                           │   │  │
│  │  │              RLBackendRegistry (注册机制)                   │   │  │
│  │  └─────────────────────────────────────────────────────────────┘   │  │
│  │                          │                                        │  │
│  │         ┌───────────────┼───────────────┐                       │  │
│  │         ▼               ▼               ▼                        │  │
│  │   ┌──────────┐   ┌──────────┐   ┌──────────┐                    │  │
│  │   │VerlBackend│   │SlimeBackend│  │  ...    │                    │  │
│  │   └──────────┘   └──────────┘   └──────────┘                    │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                     │                                       │
│                                     ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                    LoRA 权重管理层 (统一)                           │  │
│  │     - 版本化存储                                                    │  │
│  │     - 热加载通知                                                    │  │
│  │     - 云存储适配                                                    │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 组件职责矩阵

| 组件 | 位置 | 职责 | 依赖 |
|------|------|------|------|
| **Proxy** | `agent-core/openjiuwen/core/llm/gateway/proxy.py` | HTTP请求转发 + LoRA路由注入 | httpx, vLLM API |
| **Recorder** | `agent-core/openjiuwen/core/llm/gateway/recorder.py` | 轨迹录制 + Token化 | transformers (tokenizer) |
| **UnifiedTrajectory** | `agent-core/openjiuwen/core/llm/gateway/schemas.py` | 统一轨迹数据格式 | pydantic |
| **RolloutPersistence** | 复用 `agent-core/openjiuwen/dev_tools/agentrl/rollout_store/` | 轨迹持久化接口 | 现有 |
| **TrainingScheduler** | `agent-core/openjiuwen/core/llm/gateway/scheduler.py` | 训练触发调度 | asyncio |
| **TriggerPolicy** | `agent-core/openjiuwen/core/llm/gateway/trigger.py` | 自定义触发条件 | 抽象接口 |
| **RLBackend** | `agent-core/openjiuwen/core/llm/gateway/rl_backend.py` | RL后端抽象 | 抽象接口 |
| **RLBackendRegistry** | `agent-core/openjiuwen/core/llm/gateway/rl_backend.py` | 后端注册机制 | 内置 |
| **VerlBackend** | `agent-core/openjiuwen/core/llm/gateway/backends/verl_backend.py` | verl实现 | verl SDK |
| **LoRAManager** | `agent-core/openjiuwen/core/llm/gateway/lora_manager.py` | 统一权重管理 | 云存储适配 |

### 3.3 数据流

```
用户请求 → LLM Gateway (Proxy)
              │
              ├── 1. 查询 LoRAManager: 用户是否有已训练的 LoRA?
              │       ├─ 有 → 注入 lora_name 到请求
              │       └─ 无 → 直接转发基础模型
              │
              ├── 2. 转发到 LLM 推理服务
              │
              └── 3. Recorder 录制轨迹
                      ├── 3a. 提取 messages + response
                      ├── 3b. Tokenize (prompt_ids + response_ids)
                      ├── 3c. 构建 UnifiedTrajectory
                      └── 3d. 保存到 RolloutPersistence

训练调度器 (定时扫描)
    │
    ├── 4. 扫描满足触发条件的用户
    │       └─ TriggerPolicy.evaluate(user_id)
    │
    ├── 5. 收集该用户的待训练轨迹
    │
    └── 6. 提交训练任务
            ├── 6a. RLBackend.train(trajectories)
            ├── 6b. 获取训练结果 (LoRA权重路径)
            ├── 6c. LoRAManager.publish(user_id, weights_path)
            └── 6d. 通知推理服务热加载
```

---

## 4. 详细设计

### 4.1 统一轨迹格式 (UnifiedTrajectory)

**设计原则**:
- 复用 agent-core 现有的 `BaseMessage` 和 `AssistantMessage` 格式
- 录制时 Tokenize，避免重复计算
- 预留多模态扩展字段
- Reward 格式灵活可扩展

```python
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum

class RewardType(str, Enum):
    """Reward 信号来源类型"""
    PRM = "prm"              # Process Reward Model
    ENV = "env"              # 环境奖励 (如任务完成)
    HUMAN = "human"          # 人工标注
    CUSTOM = "custom"        # 自定义

@dataclass
class TokenData:
    """Token 级训练数据"""
    prompt_ids: List[int]
    response_ids: List[int]
    logprobs: List[float] = field(default_factory=list)
    loss_mask: List[int] = field(default_factory=list)
    
    # 多模态扩展 (预留)
    multimodal_tokens: Optional[Dict[str, Any]] = None

@dataclass
class TrajectoryMetadata:
    """轨迹元数据"""
    user_id: str
    session_id: str
    turn_num: int
    timestamp: str
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 2048
    tool_calls_count: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

@dataclass
class UnifiedTrajectory:
    """统一轨迹格式 - 支持多模态、灵活Reward"""
    
    # 基础标识
    trajectory_id: str
    user_id: str
    session_id: str
    turn_num: int
    
    # 输入 (BaseMessage 格式 - 天然支持多模态)
    messages: List[Dict[str, Any]]      # 原始消息列表 (OpenAI 格式)
    tools: Optional[List[Dict[str, Any]]] = None
    
    # 输出
    response: Dict[str, Any]            # LLM 响应 (content, tool_calls, etc.)
    
    # Token 级数据 (录制时 Tokenize)
    token_data: Optional[TokenData] = None
    
    # 奖励信号 (灵活扩展)
    reward: float = 0.0
    reward_type: RewardType = RewardType.CUSTOM
    reward_details: Dict[str, Any] = field(default_factory=dict)
    
    # 元数据
    metadata: Optional[TrajectoryMetadata] = None
    
    # 技能版本
    skill_version: int = 0
    
    # 预留扩展
    extra: Dict[str, Any] = field(default_factory=dict)
```

**设计辨析**:

| 选择 | 方案 | 理由 |
|------|------|------|
| Tokenize 时机 | 录制时 | 避免训练时重复计算 |
| 消息格式 | OpenAI 格式 dict | 与 LLM API 一致，天然支持多模态 |
| Reward 类型 | 枚举 + 灵活 details | 算法未定，预留扩展空间 |
| 多模态支持 | 预留 multimodal_tokens | 优先级低，先支持文本 |

### 4.2 LLM Gateway - Proxy

**职责**: HTTP 请求代理 + LoRA 路由注入

```python
import httpx
from typing import Optional, Dict, Any

class LLMProxy:
    """LLM 请求代理"""
    
    def __init__(
        self,
        upstream_url: str,           # 上游 LLM 推理服务地址
        lora_manager: "LoRAManager", # LoRA 权重管理器
        recorder: Optional["Recorder"] = None,
    ):
        self.upstream_url = upstream_url
        self.lora_manager = lora_manager
        self.recorder = recorder
        self.client = httpx.AsyncClient(timeout=120.0)
    
    async def forward_request(
        self,
        body: Dict[str, Any],
        user_id: str,
        session_id: str,
        turn_num: int,
    ) -> Dict[str, Any]:
        """转发 LLM 请求，录制轨迹"""
        
        # 1. 查询 LoRA
        lora_name = await self.lora_manager.get_active_lora(user_id)
        if lora_name:
            body = self._inject_lora(body, lora_name)
        
        # 2. 转发请求
        response = await self.client.post(
            f"{self.upstream_url}/v1/chat/completions",
            json=body,
        )
        result = response.json()
        
        # 3. 录制轨迹
        if self.recorder:
            await self.recorder.record(
                messages=body.get("messages", []),
                response=result.get("choices", [{}])[0].get("message", {}),
                user_id=user_id,
                session_id=session_id,
                turn_num=turn_num,
                model_name=body.get("model", ""),
                temperature=body.get("temperature", 0.7),
            )
        
        return result
    
    def _inject_lora(self, body: Dict[str, Any], lora_name: str) -> Dict[str, Any]:
        """注入 LoRA 路由信息"""
        extra_body = body.get("extra_body", {})
        extra_body["lora_name"] = lora_name
        body["extra_body"] = extra_body
        return body
```

**设计辨析**:

| 选择 | 方案 | 理由 |
|------|------|------|
| HTTP 客户端 | httpx | 异步支持，与 agent-core 一致 |
| LoRA 注入方式 | extra_body.lora_name | vLLM 原生支持 |
| 流式响应 | 支持 | 保持与现有 API 兼容 |

### 4.3 LLM Gateway - Recorder

**职责**: 轨迹录制 + Token化 + RolloutPersistence 写入

```python
from transformers import AutoTokenizer
from openjiuwen.dev_tools.agentrl.rollout_store.base import RolloutPersistence

class Recorder:
    """轨迹录制器"""
    
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        rollout_store: RolloutPersistence,
    ):
        self.tokenizer = tokenizer
        self.rollout_store = rollout_store
    
    async def record(
        self,
        messages: List[Dict[str, Any]],
        response: Dict[str, Any],
        user_id: str,
        session_id: str,
        turn_num: int,
        model_name: str,
        temperature: float = 0.7,
    ) -> UnifiedTrajectory:
        """录制单轮对话轨迹"""
        
        # 1. Tokenize
        prompt_text = self._build_prompt_text(messages)
        response_text = response.get("content", "")
        
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        response_ids = self.tokenizer(response_text, add_special_tokens=False)["input_ids"]
        
        # 2. 构建 TokenData
        token_data = TokenData(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            logprobs=[],  # 从 LLM 响应中提取 (如果可用)
            loss_mask=[1] * len(response_ids),
        )
        
        # 3. 构建 UnifiedTrajectory
        trajectory = UnifiedTrajectory(
            trajectory_id=uuid4().hex,
            user_id=user_id,
            session_id=session_id,
            turn_num=turn_num,
            messages=messages,
            response=response,
            token_data=token_data,
            metadata=TrajectoryMetadata(
                user_id=user_id,
                session_id=session_id,
                turn_num=turn_num,
                timestamp=datetime.utcnow().isoformat(),
                model_name=model_name,
                temperature=temperature,
            ),
        )
        
        # 4. 持久化
        await self._persist(trajectory)
        
        return trajectory
    
    def _build_prompt_text(self, messages: List[Dict[str, Any]]) -> str:
        """构建 prompt 文本"""
        # 使用 tokenizer 的 chat_template (如果可用)
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # 降级为简单拼接
            return "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    
    async def _persist(self, trajectory: UnifiedTrajectory):
        """持久化轨迹"""
        # 转换为 RolloutMessage 格式 (与现有 agentrl 兼容)
        rollout_msg = self._to_rollout_message(trajectory)
        await self.rollout_store.save_rollout(
            step=trajectory.turn_num,
            task_id=trajectory.trajectory_id,
            rollout=rollout_msg,
            phase="train",
        )
```

### 4.4 训练调度器 (TrainingScheduler)

**职责**: 定时扫描 + 触发训练

```python
import asyncio
from typing import Callable, Awaitable

class TriggerPolicy:
    """触发策略抽象接口"""
    
    async def should_trigger(self, user_id: str, pending_count: int) -> bool:
        """判断是否应该触发训练"""
        raise NotImplementedError

class ThresholdTrigger(TriggerPolicy):
    """基于数量的触发策略"""
    
    def __init__(self, threshold: int = 200):
        self.threshold = threshold
    
    async def should_trigger(self, user_id: str, pending_count: int) -> bool:
        return pending_count >= self.threshold

class TrainingScheduler:
    """训练调度器"""
    
    def __init__(
        self,
        rollout_store: RolloutPersistence,
        rl_backend: "RLBackend",
        lora_manager: "LoRAManager",
        trigger_policy: TriggerPolicy,
        scan_interval: float = 600.0,  # 10 分钟
    ):
        self.rollout_store = rollout_store
        self.rl_backend = rl_backend
        self.lora_manager = lora_manager
        self.trigger_policy = trigger_policy
        self.scan_interval = scan_interval
        self._running = False
    
    async def start(self):
        """启动调度器"""
        self._running = True
        asyncio.create_task(self._scan_loop())
    
    async def stop(self):
        """停止调度器"""
        self._running = False
    
    async def _scan_loop(self):
        """扫描循环"""
        while self._running:
            await self._scan_and_train()
            await asyncio.sleep(self.scan_interval)
    
    async def _scan_and_train(self):
        """扫描并触发训练"""
        # 1. 获取所有活跃用户
        active_users = await self._get_active_users()
        
        for user_id in active_users:
            # 2. 查询待训练轨迹数量
            pending_count = await self._get_pending_count(user_id)
            
            # 3. 判断是否触发
            if await self.trigger_policy.should_trigger(user_id, pending_count):
                await self._train_user(user_id)
    
    async def _train_user(self, user_id: str):
        """为单个用户执行训练"""
        # 1. 收集轨迹
        trajectories = await self._collect_trajectories(user_id)
        
        # 2. 执行训练
        result = await self.rl_backend.train(trajectories)
        
        # 3. 发布权重
        await self.lora_manager.publish(user_id, result.weights_path)
        
        # 4. 标记轨迹为已训练
        await self._mark_trained(user_id, trajectories)
```

**设计辨析**:

| 选择 | 方案 | 理由 |
|------|------|------|
| 触发策略 | 可插拔接口 | 算法未定，允许自定义 |
| 扫描间隔 | 可配置 | 默认10分钟，与 TARGET.md 一致 |
| 训练粒度 | 单用户顺序 | 与 TARGET.md 一致，避免并发冲突 |

### 4.5 RL 后端抽象层

**职责**: 统一 RL 训练接口，支持多后端

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Type

@dataclass
class TrainingResult:
    """训练结果"""
    user_id: str
    weights_path: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    status: str = "success"

class RLBackend(ABC):
    """RL 训练后端抽象接口"""
    
    @abstractmethod
    async def train(
        self,
        trajectories: List[UnifiedTrajectory],
        config: Optional[Dict[str, Any]] = None,
    ) -> TrainingResult:
        """执行训练，返回训练结果"""
        pass
    
    @abstractmethod
    async def get_backend_name(self) -> str:
        """返回后端名称"""
        pass

class RLBackendRegistry:
    """RL 后端注册表"""
    
    _backends: Dict[str, Type[RLBackend]] = {}
    
    @classmethod
    def register(cls, name: str):
        """装饰器注册后端"""
        def decorator(backend_cls: Type[RLBackend]) -> Type[RLBackend]:
            cls._backends[name] = backend_cls
            return backend_cls
        return decorator
    
    @classmethod
    def get_backend(cls, name: str, **kwargs) -> RLBackend:
        """获取后端实例"""
        if name not in cls._backends:
            raise ValueError(f"Unknown RL backend: {name}. Available: {list(cls._backends.keys())}")
        return cls._backends[name](**kwargs)
    
    @classmethod
    def list_backends(cls) -> List[str]:
        """列出所有已注册的后端"""
        return list(cls._backends.keys())

# 注册 verl 后端
@RLBackendRegistry.register("verl")
class VerlBackend(RLBackend):
    """Verl RL 训练后端"""
    
    async def train(self, trajectories, config=None):
        # 转换为 verl 数据格式
        # 调用 verl 训练
        # 返回训练结果
        ...
    
    async def get_backend_name(self):
        return "verl"
```

**设计辨析**:

| 选择 | 方案 | 理由 |
|------|------|------|
| 注册机制 | 装饰器 | Pythonic，易于扩展 |
| 数据转换 | 后端内部处理 | 隔离不同框架的数据格式差异 |
| 配置传递 | Dict[str, Any] | 灵活，适配不同后端 |

### 4.6 LoRA 权重管理

**职责**: 统一版本化存储 + 热加载通知

```python
class LoRAManager:
    """LoRA 权重管理器"""
    
    def __init__(
        self,
        storage_adapter: "LoRAStorageAdapter",
        inference_notifier: "InferenceNotifier",
    ):
        self.storage = storage_adapter
        self.notifier = inference_notifier
        self._active_loras: Dict[str, str] = {}  # user_id → lora_name
    
    async def get_active_lora(self, user_id: str) -> Optional[str]:
        """获取用户当前活跃的 LoRA"""
        return self._active_loras.get(user_id)
    
    async def publish(self, user_id: str, weights_path: str):
        """发布新的 LoRA 权重"""
        # 1. 生成版本号
        version = await self._next_version(user_id)
        
        # 2. 存储权重
        await self.storage.save(user_id, version, weights_path)
        
        # 3. 更新 latest 软链
        await self.storage.set_latest(user_id, version)
        
        # 4. 通知推理服务热加载
        lora_name = f"{user_id}_v{version}"
        await self.notifier.notify_update(user_id, lora_name, weights_path)
        
        # 5. 更新活跃 LoRA
        self._active_loras[user_id] = lora_name
    
    async def _next_version(self, user_id: str) -> int:
        """获取下一个版本号"""
        latest = await self.storage.get_latest_version(user_id)
        return (latest or 0) + 1

class LoRAStorageAdapter(ABC):
    """LoRA 存储适配器抽象"""
    
    @abstractmethod
    async def save(self, user_id: str, version: int, weights_path: str):
        pass
    
    @abstractmethod
    async def get_latest_version(self, user_id: str) -> Optional[int]:
        pass
    
    @abstractmethod
    async def set_latest(self, user_id: str, version: int):
        pass

class LocalStorageAdapter(LoRAStorageAdapter):
    """本地文件系统存储"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
    
    async def save(self, user_id, version, weights_path):
        dest = self.base_path / user_id / f"v{version}"
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copytree(weights_path, dest, dirs_exist_ok=True)
    
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
            json={
                "lora_name": lora_name,
                "lora_path": weights_path,
            },
        )
```

**设计辨析**:

| 选择 | 方案 | 理由 |
|------|------|------|
| 版本化 | 递增版本号 + latest 软链 | 简单可靠，vLLM 原生支持 |
| 存储适配 | 抽象接口 | 支持本地/云存储 |
| 热加载 | HTTP 通知 | vLLM 原生 API |

---

## 5. 目录结构

```
agent-core/openjiuwen/core/llm/gateway/
├── __init__.py
├── proxy.py                 # LLMProxy - HTTP 请求代理
├── recorder.py              # Recorder - 轨迹录制器
├── schemas.py               # UnifiedTrajectory, TokenData, etc.
├── scheduler.py             # TrainingScheduler - 训练调度器
├── trigger.py               # TriggerPolicy - 触发策略
├── rl_backend.py            # RLBackend, RLBackendRegistry
├── lora_manager.py          # LoRAManager, LoRAStorageAdapter, InferenceNotifier
├── backends/
│   ├── __init__.py
│   └── verl_backend.py      # VerlBackend 实现
├── stores/
│   ├── __init__.py
│   ├── local_store.py       # LocalStorageAdapter
│   └── s3_store.py          # S3StorageAdapter (预留)
└── config/
    └── gateway_config.py    # Gateway 配置
```

---

## 6. 配置设计

```yaml
# agent-core 配置文件中新增
llm_gateway:
  enabled: true
  proxy:
    upstream_url: "http://inference-service:8000"
    host: "0.0.0.0"
    port: 8001
  
  recorder:
    tokenizer_model: "Qwen/Qwen2.5-7B"
    rollout_store:
      type: "file"
      save_path: "/data/rollouts"
      flush_interval: 100
  
  scheduler:
    enabled: true
    scan_interval: 600  # 10 分钟
    trigger:
      type: "threshold"
      threshold: 200
  
  rl_backend:
    name: "verl"
    config:
      # verl 特定配置
      ppo_epochs: 4
      learning_rate: 1e-6
  
  lora:
    storage:
      type: "local"
      base_path: "/data/lora_weights"
    inference:
      url: "http://inference-service:8000"
```

---

## 7. 与 jiuwenclaw 的集成

### 7.1 集成方式

jiuwenclaw **无需修改代码**，只需配置使用 LLM Gateway：

```yaml
# jiuwenclaw 配置文件
agent:
  llm:
    # 原来: 直接调用 LLM API
    # api_base: "https://api.openai.com/v1"
    
    # 现在: 通过 LLM Gateway
    api_base: "http://localhost:8001/v1"
```

### 7.2 影响分析

| 组件 | 是否需要修改 | 说明 |
|------|-------------|------|
| jiuwenclaw/gateway/ | ❌ 否 | 消息网关职责不变 |
| jiuwenclaw/agentserver/ | ❌ 否 | 只需更改 LLM API 地址 |
| agent-core/llm/gateway/ | ✅ 新增 | 在线RL 模块 |

---

## 8. 设计决策汇总

| 决策点 | 选择 | 理由 |
|--------|------|------|
| LLM Gateway 位置 | agent-core | 泛化性，可服务于任何上层应用 |
| 训练调度器位置 | agent-core | 与 Gateway 同层，保持低耦合 |
| 轨迹存储 | 复用 RolloutPersistence | 与离线RL数据格式一致 |
| Tokenize 时机 | 录制时 | 避免重复计算 |
| Reward 格式 | 灵活 Dict | 算法未定，预留扩展 |
| RL 后端 | 可插拔注册 | 支持 verl/slime 等多框架 |
| LoRA 管理 | 统一抽象层 | 对接不同云存储平台 |
| 触发策略 | 可插拔接口 | 允许自定义触发条件 |
| 多模态 | 预留扩展 | 优先级低，先支持文本 |

---

## 9. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| Tokenizer 与推理服务不一致 | 训练数据偏差 | 使用相同模型版本的 tokenizer |
| 训练时 GPU 资源不足 | 训练失败 | 调度器检测资源可用性 |
| LoRA 热加载失败 | 用户使用旧权重 | 通知器重试 + 回滚机制 |
| 轨迹数据量过大 | 存储压力 | 定期归档 + 压缩 |

---

## 10. 后续工作

1. **详细实现计划**: 基于此设计文档，编写分步实现计划
2. **RL 算法对接**: 与算法团队确认 verl 后端的具体接口
3. **Reward 计算**: 设计 LLM-as-Judge 或环境 Reward 的具体实现
4. **性能测试**: 验证 Gateway 代理的延迟影响
5. **多模态扩展**: 后续支持图片、音频等模态

---

> **文档结束**
> 
> 本文档为 Jiuwen agent-core 在线强化学习框架的设计蓝图。
> 所有设计决策均基于对 agent-core、jiuwenclaw、openclaw-rl、MetaClaw 四个项目的深入分析。
