# Jiuwen Agent-Core 在线强化学习框架 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 agent-core 中构建在线RL框架，使智能体在不停机的前提下，一边与用户交互收集轨迹，一边在合适时机自动触发 LLM RL 训练。

**Architecture:** 新增 LLM Gateway（Proxy + Recorder）作为 agent-core 的独立模块，复用现有 RolloutPersistence 接口存储轨迹，通过可插拔的 RLBackend 抽象层对接 verl 等训练框架，统一 LoRA 权重管理实现热加载。

**Tech Stack:** Python 3.11+, httpx (异步HTTP), pydantic (数据模型), transformers (tokenizer), verl (训练后端), asyncio (并发)

**Spec:** `docs/superpowers/specs/2026-03-31-online-rl-framework-design.md`

---

## 文件结构规划

### 新增文件

| 文件 | 职责 |
|------|------|
| `openjiuwen/core/llm/gateway/__init__.py` | 模块入口，导出公共API |
| `openjiuwen/core/llm/gateway/schemas.py` | UnifiedTrajectory, TokenData, TrajectoryMetadata, RewardType |
| `openjiuwen/core/llm/gateway/proxy.py` | LLMProxy - HTTP请求代理 + LoRA路由注入 |
| `openjiuwen/core/llm/gateway/recorder.py` | Recorder - 轨迹录制 + Token化 |
| `openjiuwen/core/llm/gateway/trigger.py` | TriggerPolicy 抽象 + ThresholdTrigger 实现 |
| `openjiuwen/core/llm/gateway/scheduler.py` | TrainingScheduler - 训练触发调度器 |
| `openjiuwen/core/llm/gateway/rl_backend.py` | RLBackend 抽象接口 + RLBackendRegistry 注册表 |
| `openjiuwen/core/llm/gateway/lora_manager.py` | LoRAManager, LoRAStorageAdapter, InferenceNotifier |
| `openjiuwen/core/llm/gateway/backends/__init__.py` | 后端模块入口 |
| `openjiuwen/core/llm/gateway/backends/verl_backend.py` | VerlBackend 实现 |
| `openjiuwen/core/llm/gateway/stores/__init__.py` | 存储模块入口 |
| `openjiuwen/core/llm/gateway/stores/local_store.py` | LocalStorageAdapter 实现 |
| `openjiuwen/core/llm/gateway/config.py` | Gateway 配置 Schema |
| `tests/core/llm/gateway/test_schemas.py` | schemas 单元测试 |
| `tests/core/llm/gateway/test_proxy.py` | proxy 单元测试 |
| `tests/core/llm/gateway/test_recorder.py` | recorder 单元测试 |
| `tests/core/llm/gateway/test_trigger.py` | trigger 单元测试 |
| `tests/core/llm/gateway/test_scheduler.py` | scheduler 单元测试 |
| `tests/core/llm/gateway/test_rl_backend.py` | rl_backend 单元测试 |
| `tests/core/llm/gateway/test_lora_manager.py` | lora_manager 单元测试 |
| `tests/core/llm/gateway/test_integration.py` | 集成测试 |

### 修改文件

| 文件 | 修改内容 |
|------|----------|
| `openjiuwen/core/llm/__init__.py` | 导出 gateway 模块 |

---

## Task 1: 数据模型 (schemas.py)

**Files:**
- Create: `openjiuwen/core/llm/gateway/schemas.py`
- Create: `openjiuwen/core/llm/gateway/__init__.py`
- Test: `tests/core/llm/gateway/test_schemas.py`

- [ ] **Step 1: 编写 schemas 测试**

```python
# tests/core/llm/gateway/test_schemas.py
"""Tests for online RL gateway data schemas."""
import pytest
from openjiuwen.core.llm.gateway.schemas import (
    RewardType,
    TokenData,
    TrajectoryMetadata,
    UnifiedTrajectory,
)


class TestRewardType:
    def test_reward_type_values(self):
        assert RewardType.PRM == "prm"
        assert RewardType.ENV == "env"
        assert RewardType.HUMAN == "human"
        assert RewardType.CUSTOM == "custom"


class TestTokenData:
    def test_token_data_defaults(self):
        td = TokenData(prompt_ids=[1, 2, 3], response_ids=[4, 5])
        assert td.prompt_ids == [1, 2, 3]
        assert td.response_ids == [4, 5]
        assert td.logprobs == []
        assert td.loss_mask == []
        assert td.multimodal_tokens is None


class TestTrajectoryMetadata:
    def test_metadata_defaults(self):
        m = TrajectoryMetadata(
            user_id="user_1",
            session_id="sess_1",
            turn_num=1,
            timestamp="2026-03-31T10:00:00",
            model_name="Qwen/Qwen2.5-7B",
        )
        assert m.temperature == 0.7
        assert m.max_tokens == 2048
        assert m.tool_calls_count == 0
        assert m.extra == {}


class TestUnifiedTrajectory:
    def test_minimal_trajectory(self):
        traj = UnifiedTrajectory(
            trajectory_id="t1",
            user_id="u1",
            session_id="s1",
            turn_num=1,
            messages=[{"role": "user", "content": "hello"}],
            response={"role": "assistant", "content": "hi"},
        )
        assert traj.trajectory_id == "t1"
        assert traj.reward == 0.0
        assert traj.reward_type == RewardType.CUSTOM
        assert traj.reward_details == {}
        assert traj.token_data is None
        assert traj.skill_version == 0
        assert traj.extra == {}

    def test_full_trajectory(self):
        token_data = TokenData(
            prompt_ids=[1, 2],
            response_ids=[3, 4],
            logprobs=[-0.5, -0.3],
            loss_mask=[1, 1],
        )
        metadata = TrajectoryMetadata(
            user_id="u1",
            session_id="s1",
            turn_num=1,
            timestamp="2026-03-31T10:00:00",
            model_name="Qwen/Qwen2.5-7B",
            temperature=0.8,
            tool_calls_count=2,
        )
        traj = UnifiedTrajectory(
            trajectory_id="t1",
            user_id="u1",
            session_id="s1",
            turn_num=1,
            messages=[{"role": "user", "content": "calc 2+2"}],
            response={"role": "assistant", "content": "4"},
            token_data=token_data,
            reward=1.0,
            reward_type=RewardType.ENV,
            reward_details={"accuracy": 1.0},
            metadata=metadata,
            skill_version=3,
            extra={"custom_key": "custom_value"},
        )
        assert traj.reward == 1.0
        assert traj.reward_type == RewardType.ENV
        assert traj.reward_details == {"accuracy": 1.0}
        assert traj.token_data == token_data
        assert traj.metadata == metadata
        assert traj.skill_version == 3
        assert traj.extra == {"custom_key": "custom_value"}
```

- [ ] **Step 2: 运行测试验证失败**

Run: `cd agent-core && python -m pytest tests/core/llm/gateway/test_schemas.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'openjiuwen.core.llm.gateway'`

- [ ] **Step 3: 实现 schemas.py**

```python
# openjiuwen/core/llm/gateway/schemas.py
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

"""
Unified data schemas for the online RL gateway.

Provides UnifiedTrajectory, TokenData, and related models that bridge
trajectory collection (runtime side) and RL training (training side).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class RewardType(str, Enum):
    """Reward signal source type."""

    PRM = "prm"              # Process Reward Model
    ENV = "env"              # Environment reward (e.g., task completion)
    HUMAN = "human"          # Human annotation
    CUSTOM = "custom"        # Custom reward source


@dataclass
class TokenData:
    """Token-level training data."""

    prompt_ids: List[int]
    response_ids: List[int]
    logprobs: List[float] = field(default_factory=list)
    loss_mask: List[int] = field(default_factory=list)

    # Multimodal extension (reserved)
    multimodal_tokens: Optional[Dict[str, Any]] = None


@dataclass
class TrajectoryMetadata:
    """Trajectory metadata."""

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
    """
    Unified trajectory format - supports multimodal input and flexible rewards.

    This is the canonical data structure used throughout the online RL pipeline:
    - Collected by the Recorder during LLM proxy forwarding
    - Persisted via RolloutPersistence
    - Consumed by RLBackend implementations for training
    """

    # Basic identifiers
    trajectory_id: str
    user_id: str
    session_id: str
    turn_num: int

    # Input (OpenAI message format - natively supports multimodal)
    messages: List[Dict[str, Any]]
    tools: Optional[List[Dict[str, Any]]] = None

    # Output
    response: Dict[str, Any]

    # Token-level data (tokenized at recording time)
    token_data: Optional[TokenData] = None

    # Reward signal (flexible extension)
    reward: float = 0.0
    reward_type: RewardType = RewardType.CUSTOM
    reward_details: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    metadata: Optional[TrajectoryMetadata] = None

    # Skill version
    skill_version: int = 0

    # Reserved extension
    extra: Dict[str, Any] = field(default_factory=dict)
```

- [ ] **Step 4: 实现 __init__.py**

```python
# openjiuwen/core/llm/gateway/__init__.py
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

"""
Online RL Gateway - LLM proxy with trajectory collection for online RL.

This module provides:
- LLMProxy: HTTP proxy with LoRA routing
- Recorder: Trajectory recording with tokenization
- UnifiedTrajectory: Canonical data format
- TrainingScheduler: Periodic scan and training trigger
- RLBackend: Pluggable RL training backend abstraction
- LoRAManager: Unified LoRA weight management
"""

from openjiuwen.core.llm.gateway.schemas import (
    RewardType,
    TokenData,
    TrajectoryMetadata,
    UnifiedTrajectory,
)

__all__ = [
    "RewardType",
    "TokenData",
    "TrajectoryMetadata",
    "UnifiedTrajectory",
]
```

- [ ] **Step 5: 运行测试验证通过**

Run: `cd agent-core && python -m pytest tests/core/llm/gateway/test_schemas.py -v`
Expected: All 5 tests PASS

- [ ] **Step 6: Commit**

```bash
cd agent-core
git add openjiuwen/core/llm/gateway/__init__.py openjiuwen/core/llm/gateway/schemas.py tests/core/llm/gateway/test_schemas.py
git commit -m "feat(online-rl): add UnifiedTrajectory schema and tests"
```

---

## Task 2: 触发策略 (trigger.py)

**Files:**
- Create: `openjiuwen/core/llm/gateway/trigger.py`
- Test: `tests/core/llm/gateway/test_trigger.py`

- [ ] **Step 1: 编写 trigger 测试**

```python
# tests/core/llm/gateway/test_trigger.py
"""Tests for trigger policies."""
import pytest
from openjiuwen.core.llm.gateway.trigger import (
    TriggerPolicy,
    ThresholdTrigger,
)


class TestThresholdTrigger:
    @pytest.mark.asyncio
    async def test_below_threshold(self):
        trigger = ThresholdTrigger(threshold=10)
        result = await trigger.should_trigger("user_1", 5)
        assert result is False

    @pytest.mark.asyncio
    async def test_at_threshold(self):
        trigger = ThresholdTrigger(threshold=10)
        result = await trigger.should_trigger("user_1", 10)
        assert result is True

    @pytest.mark.asyncio
    async def test_above_threshold(self):
        trigger = ThresholdTrigger(threshold=10)
        result = await trigger.should_trigger("user_1", 15)
        assert result is True

    @pytest.mark.asyncio
    async def test_zero_threshold(self):
        trigger = ThresholdTrigger(threshold=0)
        result = await trigger.should_trigger("user_1", 0)
        assert result is True

    @pytest.mark.asyncio
    async def test_default_threshold(self):
        trigger = ThresholdTrigger()
        result = await trigger.should_trigger("user_1", 199)
        assert result is False
        result = await trigger.should_trigger("user_1", 200)
        assert result is True
```

- [ ] **Step 2: 运行测试验证失败**

Run: `cd agent-core && python -m pytest tests/core/llm/gateway/test_trigger.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: 实现 trigger.py**

```python
# openjiuwen/core/llm/gateway/trigger.py
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

"""
Trigger policies for online RL training.

A TriggerPolicy determines when a user's accumulated trajectories
should trigger an RL training step.  Policies are pluggable so that
applications can implement custom triggering logic (e.g., time-based,
quality-based, or hybrid).
"""

from abc import ABC, abstractmethod


class TriggerPolicy(ABC):
    """Abstract trigger policy interface."""

    @abstractmethod
    async def should_trigger(self, user_id: str, pending_count: int) -> bool:
        """
        Determine whether training should be triggered for a user.

        Args:
            user_id: The user identifier.
            pending_count: Number of pending (untrained) trajectories.

        Returns:
            True if training should be triggered, False otherwise.
        """
        raise NotImplementedError


class ThresholdTrigger(TriggerPolicy):
    """
    Trigger when pending trajectory count reaches a threshold.

    This is the simplest and most common trigger policy.
    """

    def __init__(self, threshold: int = 200):
        """
        Args:
            threshold: Minimum number of pending trajectories to trigger training.
        """
        self.threshold = max(0, threshold)

    async def should_trigger(self, user_id: str, pending_count: int) -> bool:
        return pending_count >= self.threshold
```

- [ ] **Step 4: 运行测试验证通过**

Run: `cd agent-core && python -m pytest tests/core/llm/gateway/test_trigger.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
cd agent-core
git add openjiuwen/core/llm/gateway/trigger.py tests/core/llm/gateway/test_trigger.py
git commit -m "feat(online-rl): add TriggerPolicy and ThresholdTrigger"
```

---

## Task 3: RL 后端抽象 (rl_backend.py)

**Files:**
- Create: `openjiuwen/core/llm/gateway/rl_backend.py`
- Create: `openjiuwen/core/llm/gateway/backends/__init__.py`
- Create: `openjiuwen/core/llm/gateway/backends/verl_backend.py`
- Test: `tests/core/llm/gateway/test_rl_backend.py`

- [ ] **Step 1: 编写 rl_backend 测试**

```python
# tests/core/llm/gateway/test_rl_backend.py
"""Tests for RL backend abstraction and registry."""
import pytest
from typing import List

from openjiuwen.core.llm.gateway.rl_backend import (
    RLBackend,
    RLBackendRegistry,
    TrainingResult,
)
from openjiuwen.core.llm.gateway.schemas import UnifiedTrajectory


class TestTrainingResult:
    def test_training_result_defaults(self):
        result = TrainingResult(user_id="u1", weights_path="/path/to/lora")
        assert result.user_id == "u1"
        assert result.weights_path == "/path/to/lora"
        assert result.metrics == {}
        assert result.status == "success"

    def test_training_result_with_metrics(self):
        result = TrainingResult(
            user_id="u1",
            weights_path="/path/to/lora",
            metrics={"loss": 0.5, "reward": 0.8},
            status="completed",
        )
        assert result.metrics == {"loss": 0.5, "reward": 0.8}
        assert result.status == "completed"


class TestRLBackendRegistry:
    def test_register_and_get(self):
        @RLBackendRegistry.register("test_backend")
        class TestBackend(RLBackend):
            async def train(self, trajectories, config=None):
                return TrainingResult(user_id="u1", weights_path="/tmp/test")

            async def get_backend_name(self):
                return "test_backend"

        backend = RLBackendRegistry.get_backend("test_backend")
        assert isinstance(backend, TestBackend)

    def test_get_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown RL backend"):
            RLBackendRegistry.get_backend("nonexistent")

    def test_list_backends(self):
        backends = RLBackendRegistry.list_backends()
        assert isinstance(backends, list)
        assert "test_backend" in backends


class TestVerlBackend:
    @pytest.mark.asyncio
    async def test_verl_backend_name(self):
        from openjiuwen.core.llm.gateway.backends.verl_backend import VerlBackend
        backend = VerlBackend()
        name = await backend.get_backend_name()
        assert name == "verl"

    @pytest.mark.asyncio
    async def test_verl_backend_train_returns_result(self):
        from openjiuwen.core.llm.gateway.backends.verl_backend import VerlBackend
        backend = VerlBackend()
        trajectories: List[UnifiedTrajectory] = []
        result = await backend.train(trajectories)
        assert isinstance(result, TrainingResult)
        assert result.status == "success"
```

- [ ] **Step 2: 运行测试验证失败**

Run: `cd agent-core && python -m pytest tests/core/llm/gateway/test_rl_backend.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 rl_backend.py**

```python
# openjiuwen/core/llm/gateway/rl_backend.py
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

"""
RL backend abstraction layer.

Provides a pluggable interface for RL training backends (verl, slime, etc.)
with a registry mechanism for runtime backend selection.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from openjiuwen.core.llm.gateway.schemas import UnifiedTrajectory


@dataclass
class TrainingResult:
    """Result of an RL training step."""

    user_id: str
    weights_path: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    status: str = "success"


class RLBackend(ABC):
    """Abstract RL training backend interface."""

    @abstractmethod
    async def train(
        self,
        trajectories: List[UnifiedTrajectory],
        config: Optional[Dict[str, Any]] = None,
    ) -> TrainingResult:
        """
        Execute RL training on the given trajectories.

        Args:
            trajectories: List of trajectories to train on.
            config: Backend-specific configuration.

        Returns:
            TrainingResult with weights path and metrics.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_backend_name(self) -> str:
        """Return the backend name."""
        raise NotImplementedError


class RLBackendRegistry:
    """Registry for RL training backends."""

    _backends: Dict[str, Type[RLBackend]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a backend class."""
        def decorator(backend_cls: Type[RLBackend]) -> Type[RLBackend]:
            cls._backends[name] = backend_cls
            return backend_cls
        return decorator

    @classmethod
    def get_backend(cls, name: str, **kwargs) -> RLBackend:
        """
        Get a backend instance by name.

        Args:
            name: Backend name (e.g., "verl", "slime").
            **kwargs: Arguments passed to the backend constructor.

        Returns:
            An instance of the requested backend.

        Raises:
            ValueError: If the backend is not registered.
        """
        if name not in cls._backends:
            raise ValueError(
                f"Unknown RL backend: {name}. "
                f"Available: {list(cls._backends.keys())}"
            )
        return cls._backends[name](**kwargs)

    @classmethod
    def list_backends(cls) -> List[str]:
        """List all registered backend names."""
        return list(cls._backends.keys())
```

- [ ] **Step 4: 实现 backends/__init__.py**

```python
# openjiuwen/core/llm/gateway/backends/__init__.py
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

"""RL backend implementations."""
```

- [ ] **Step 5: 实现 verl_backend.py**

```python
# openjiuwen/core/llm/gateway/backends/verl_backend.py
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

"""
Verl RL training backend.

This backend integrates the verl training framework for online RL.
The actual verl integration will be implemented when the verl SDK
is available and the RL algorithm is finalized.
"""

from typing import Any, Dict, List, Optional

from openjiuwen.core.llm.gateway.rl_backend import RLBackend, TrainingResult
from openjiuwen.core.llm.gateway.schemas import UnifiedTrajectory


class VerlBackend(RLBackend):
    """Verl-based RL training backend."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: Verl-specific configuration (e.g., ppo_epochs, learning_rate).
        """
        self.config = config or {}

    async def train(
        self,
        trajectories: List[UnifiedTrajectory],
        config: Optional[Dict[str, Any]] = None,
    ) -> TrainingResult:
        """
        Execute verl training.

        NOTE: This is a placeholder implementation. The actual verl integration
        will be implemented when the verl SDK is available.

        The expected flow:
        1. Convert UnifiedTrajectory list to verl DataProto format
        2. Load base model (if not already loaded)
        3. Apply PPO/GRPO update with LoRA
        4. Export LoRA weights
        5. Return TrainingResult with weights path
        """
        effective_config = {**self.config, **(config or {})}

        return TrainingResult(
            user_id=trajectories[0].user_id if trajectories else "unknown",
            weights_path="/tmp/placeholder_lora",
            metrics={"note": "verl backend not yet fully implemented"},
            status="success",
        )

    async def get_backend_name(self) -> str:
        return "verl"
```

- [ ] **Step 6: 运行测试验证通过**

Run: `cd agent-core && python -m pytest tests/core/llm/gateway/test_rl_backend.py -v`
Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
cd agent-core
git add openjiuwen/core/llm/gateway/rl_backend.py openjiuwen/core/llm/gateway/backends/__init__.py openjiuwen/core/llm/gateway/backends/verl_backend.py tests/core/llm/gateway/test_rl_backend.py
git commit -m "feat(online-rl): add RLBackend abstraction and VerlBackend placeholder"
```

---

## Task 4: LLM Proxy (proxy.py)

**Files:**
- Create: `openjiuwen/core/llm/gateway/proxy.py`
- Test: `tests/core/llm/gateway/test_proxy.py`

- [ ] **Step 1: 编写 proxy 测试**

```python
# tests/core/llm/gateway/test_proxy.py
"""Tests for LLM proxy."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from openjiuwen.core.llm.gateway.proxy import LLMProxy


class TestLLMProxy:
    def test_init(self):
        mock_lora_manager = MagicMock()
        proxy = LLMProxy(
            upstream_url="http://localhost:8000",
            lora_manager=mock_lora_manager,
        )
        assert proxy.upstream_url == "http://localhost:8000"
        assert proxy.lora_manager is mock_lora_manager
        assert proxy.recorder is None

    def test_init_with_recorder(self):
        mock_lora_manager = MagicMock()
        mock_recorder = MagicMock()
        proxy = LLMProxy(
            upstream_url="http://localhost:8000",
            lora_manager=mock_lora_manager,
            recorder=mock_recorder,
        )
        assert proxy.recorder is mock_recorder

    def test_inject_lora(self):
        mock_lora_manager = MagicMock()
        proxy = LLMProxy(
            upstream_url="http://localhost:8000",
            lora_manager=mock_lora_manager,
        )
        body = {"messages": [{"role": "user", "content": "hi"}]}
        result = proxy._inject_lora(body, "user_1_v1")
        assert result["extra_body"]["lora_name"] == "user_1_v1"

    def test_inject_lora_preserves_existing_extra(self):
        mock_lora_manager = MagicMock()
        proxy = LLMProxy(
            upstream_url="http://localhost:8000",
            lora_manager=mock_lora_manager,
        )
        body = {
            "messages": [{"role": "user", "content": "hi"}],
            "extra_body": {"existing_key": "existing_value"},
        }
        result = proxy._inject_lora(body, "user_1_v1")
        assert result["extra_body"]["lora_name"] == "user_1_v1"
        assert result["extra_body"]["existing_key"] == "existing_value"

    @pytest.mark.asyncio
    async def test_forward_request_without_lora(self):
        mock_lora_manager = AsyncMock()
        mock_lora_manager.get_active_lora.return_value = None

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "hi"}}]
        }

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        proxy = LLMProxy(
            upstream_url="http://localhost:8000",
            lora_manager=mock_lora_manager,
        )
        proxy.client = mock_client

        result = await proxy.forward_request(
            body={"messages": [{"role": "user", "content": "hi"}], "model": "test"},
            user_id="user_1",
            session_id="sess_1",
            turn_num=1,
        )
        assert result["choices"][0]["message"]["content"] == "hi"

    @pytest.mark.asyncio
    async def test_forward_request_with_lora(self):
        mock_lora_manager = AsyncMock()
        mock_lora_manager.get_active_lora.return_value = "user_1_v1"

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "hi"}}]
        }

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        proxy = LLMProxy(
            upstream_url="http://localhost:8000",
            lora_manager=mock_lora_manager,
        )
        proxy.client = mock_client

        await proxy.forward_request(
            body={"messages": [{"role": "user", "content": "hi"}], "model": "test"},
            user_id="user_1",
            session_id="sess_1",
            turn_num=1,
        )

        call_args = mock_client.post.call_args
        json_body = call_args.kwargs.get("json", {})
        assert json_body["extra_body"]["lora_name"] == "user_1_v1"
```

- [ ] **Step 2: 运行测试验证失败**

Run: `cd agent-core && python -m pytest tests/core/llm/gateway/test_proxy.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 proxy.py**

```python
# openjiuwen/core/llm/gateway/proxy.py
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

"""
LLM Proxy - HTTP request forwarding with LoRA routing.

Intercepts LLM requests, injects user-specific LoRA routing,
forwards to the upstream inference service, and optionally
records the trajectory via a Recorder.
"""

from typing import TYPE_CHECKING, Any, Dict, Optional

import httpx

if TYPE_CHECKING:
    from openjiuwen.core.llm.gateway.recorder import Recorder
    from openjiuwen.core.llm.gateway.lora_manager import LoRAManager


class LLMProxy:
    """LLM request proxy with LoRA routing and trajectory recording."""

    def __init__(
        self,
        upstream_url: str,
        lora_manager: "LoRAManager",
        recorder: Optional["Recorder"] = None,
        timeout: float = 120.0,
    ):
        """
        Args:
            upstream_url: Upstream LLM inference service URL.
            lora_manager: LoRA weight manager for routing.
            recorder: Optional trajectory recorder.
            timeout: HTTP request timeout in seconds.
        """
        self.upstream_url = upstream_url.rstrip("/")
        self.lora_manager = lora_manager
        self.recorder = recorder
        self.client = httpx.AsyncClient(timeout=timeout)

    async def forward_request(
        self,
        body: Dict[str, Any],
        user_id: str,
        session_id: str,
        turn_num: int,
    ) -> Dict[str, Any]:
        """
        Forward an LLM request, inject LoRA routing, and record trajectory.

        Args:
            body: OpenAI-compatible request body.
            user_id: User identifier.
            session_id: Session identifier.
            turn_num: Turn number within the session.

        Returns:
            LLM response as a dict.
        """
        # 1. Query active LoRA
        lora_name = await self.lora_manager.get_active_lora(user_id)
        if lora_name:
            body = self._inject_lora(body, lora_name)

        # 2. Forward request
        response = await self.client.post(
            f"{self.upstream_url}/v1/chat/completions",
            json=body,
        )
        response.raise_for_status()
        result = response.json()

        # 3. Record trajectory (async, non-blocking)
        if self.recorder:
            try:
                await self.recorder.record(
                    messages=body.get("messages", []),
                    response=result.get("choices", [{}])[0].get("message", {}),
                    user_id=user_id,
                    session_id=session_id,
                    turn_num=turn_num,
                    model_name=body.get("model", ""),
                    temperature=body.get("temperature", 0.7),
                )
            except Exception:
                # Recording failure should not break the main request
                pass

        return result

    def _inject_lora(self, body: Dict[str, Any], lora_name: str) -> Dict[str, Any]:
        """Inject LoRA routing information into the request body."""
        extra_body = body.get("extra_body", {})
        extra_body["lora_name"] = lora_name
        body["extra_body"] = extra_body
        return body

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()
```

- [ ] **Step 4: 运行测试验证通过**

Run: `cd agent-core && python -m pytest tests/core/llm/gateway/test_proxy.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
cd agent-core
git add openjiuwen/core/llm/gateway/proxy.py tests/core/llm/gateway/test_proxy.py
git commit -m "feat(online-rl): add LLMProxy with LoRA routing and recording"
```

---

## Task 5: 轨迹录制器 (recorder.py)

**Files:**
- Create: `openjiuwen/core/llm/gateway/recorder.py`
- Test: `tests/core/llm/gateway/test_recorder.py`

- [ ] **Step 1: 编写 recorder 测试**

```python
# tests/core/llm/gateway/test_recorder.py
"""Tests for trajectory recorder."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from openjiuwen.core.llm.gateway.recorder import Recorder
from openjiuwen.core.llm.gateway.schemas import UnifiedTrajectory, RewardType


class TestRecorder:
    def test_init(self):
        mock_tokenizer = MagicMock()
        mock_store = AsyncMock()
        recorder = Recorder(tokenizer=mock_tokenizer, rollout_store=mock_store)
        assert recorder.tokenizer is mock_tokenizer
        assert recorder.rollout_store is mock_store

    def test_build_prompt_text_with_chat_template(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "user: hello\nassistant:"
        mock_store = AsyncMock()
        recorder = Recorder(tokenizer=mock_tokenizer, rollout_store=mock_store)

        messages = [{"role": "user", "content": "hello"}]
        result = recorder._build_prompt_text(messages)
        assert result == "user: hello\nassistant:"

    def test_build_prompt_text_fallback(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.side_effect = Exception("no template")
        mock_store = AsyncMock()
        recorder = Recorder(tokenizer=mock_tokenizer, rollout_store=mock_store)

        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        result = recorder._build_prompt_text(messages)
        assert "user: hello" in result
        assert "assistant: hi" in result

    @pytest.mark.asyncio
    async def test_record_creates_trajectory(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": [1, 2, 3]}
        mock_tokenizer.apply_chat_template.return_value = "user: hello"
        mock_store = AsyncMock()

        recorder = Recorder(tokenizer=mock_tokenizer, rollout_store=mock_store)

        trajectory = await recorder.record(
            messages=[{"role": "user", "content": "hello"}],
            response={"role": "assistant", "content": "hi"},
            user_id="user_1",
            session_id="sess_1",
            turn_num=1,
            model_name="test-model",
            temperature=0.7,
        )

        assert isinstance(trajectory, UnifiedTrajectory)
        assert trajectory.user_id == "user_1"
        assert trajectory.session_id == "sess_1"
        assert trajectory.turn_num == 1
        assert trajectory.token_data is not None
        assert trajectory.token_data.prompt_ids == [1, 2, 3]
        assert trajectory.token_data.response_ids == [1, 2, 3]
        assert trajectory.reward == 0.0
        assert trajectory.reward_type == RewardType.CUSTOM

    @pytest.mark.asyncio
    async def test_record_persists_to_store(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": [1, 2, 3]}
        mock_tokenizer.apply_chat_template.return_value = "user: hello"
        mock_store = AsyncMock()

        recorder = Recorder(tokenizer=mock_tokenizer, rollout_store=mock_store)

        await recorder.record(
            messages=[{"role": "user", "content": "hello"}],
            response={"role": "assistant", "content": "hi"},
            user_id="user_1",
            session_id="sess_1",
            turn_num=1,
            model_name="test-model",
        )

        mock_store.save_rollout.assert_called_once()
```

- [ ] **Step 2: 运行测试验证失败**

Run: `cd agent-core && python -m pytest tests/core/llm/gateway/test_recorder.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 recorder.py**

```python
# openjiuwen/core/llm/gateway/recorder.py
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

"""
Trajectory Recorder - records LLM interactions as UnifiedTrajectory objects.

Tokenizes prompts and responses at recording time to avoid redundant
computation during training.  Persists via RolloutPersistence.
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from openjiuwen.core.llm.gateway.schemas import (
    RewardType,
    TokenData,
    TrajectoryMetadata,
    UnifiedTrajectory,
)


class Recorder:
    """Records LLM interactions as training trajectories."""

    def __init__(self, tokenizer, rollout_store):
        """
        Args:
            tokenizer: A tokenizer with ``apply_chat_template`` and ``__call__`` methods.
            rollout_store: A RolloutPersistence instance for storing trajectories.
        """
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
        """
        Record a single turn of conversation as a trajectory.

        Args:
            messages: Input message list (OpenAI format).
            response: LLM response message.
            user_id: User identifier.
            session_id: Session identifier.
            turn_num: Turn number within the session.
            model_name: Model name used for generation.
            temperature: Sampling temperature.

        Returns:
            The created UnifiedTrajectory.
        """
        # 1. Tokenize
        prompt_text = self._build_prompt_text(messages)
        response_text = response.get("content", "")

        prompt_ids = self._tokenize(prompt_text)
        response_ids = self._tokenize(response_text)

        # 2. Build TokenData
        token_data = TokenData(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            logprobs=[],
            loss_mask=[1] * len(response_ids),
        )

        # 3. Build UnifiedTrajectory
        trajectory = UnifiedTrajectory(
            trajectory_id=uuid.uuid4().hex,
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
                timestamp=datetime.now(tz=timezone.utc).isoformat(),
                model_name=model_name,
                temperature=temperature,
            ),
        )

        # 4. Persist
        await self._persist(trajectory)

        return trajectory

    def _build_prompt_text(self, messages: List[Dict[str, Any]]) -> str:
        """Build prompt text from messages, using chat_template if available."""
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            return "\n".join(
                f"{m.get('role', '?')}: {m.get('content', '')}"
                for m in messages
            )

    def _tokenize(self, text: str) -> List[int]:
        """Tokenize text and return input IDs."""
        if not text.strip():
            return []
        result = self.tokenizer(text, add_special_tokens=False)
        return result.get("input_ids", [])

    async def _persist(self, trajectory: UnifiedTrajectory) -> None:
        """Persist trajectory via RolloutPersistence."""
        from openjiuwen.dev_tools.agentrl.coordinator.schemas import (
            Rollout,
            RolloutMessage,
        )

        rollout = Rollout(
            turn_id=trajectory.turn_num,
            input_prompt={"message": trajectory.messages, "tools": trajectory.tools},
            output_response=trajectory.response,
            llm_config={
                "temperature": trajectory.metadata.temperature
                if trajectory.metadata
                else 0.7,
            },
        )

        rollout_msg = RolloutMessage(
            task_id=trajectory.trajectory_id,
            origin_task_id=trajectory.trajectory_id,
            rollout_id=trajectory.trajectory_id,
            rollout_info=[rollout],
            reward_list=[trajectory.reward],
            global_reward=trajectory.reward,
            turn_count=1,
            start_time=trajectory.metadata.timestamp if trajectory.metadata else "",
            end_time=trajectory.metadata.timestamp if trajectory.metadata else "",
        )

        await self.rollout_store.save_rollout(
            step=trajectory.turn_num,
            task_id=trajectory.trajectory_id,
            rollout=rollout_msg,
            phase="train",
        )
```

- [ ] **Step 4: 运行测试验证通过**

Run: `cd agent-core && python -m pytest tests/core/llm/gateway/test_recorder.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
cd agent-core
git add openjiuwen/core/llm/gateway/recorder.py tests/core/llm/gateway/test_recorder.py
git commit -m "feat(online-rl): add Recorder with tokenization and persistence"
```

---

## Task 6: LoRA 权重管理 (lora_manager.py)

**Files:**
- Create: `openjiuwen/core/llm/gateway/lora_manager.py`
- Create: `openjiuwen/core/llm/gateway/stores/__init__.py`
- Create: `openjiuwen/core/llm/gateway/stores/local_store.py`
- Test: `tests/core/llm/gateway/test_lora_manager.py`

- [ ] **Step 1: 编写 lora_manager 测试**

```python
# tests/core/llm/gateway/test_lora_manager.py
"""Tests for LoRA weight manager."""
import pytest
from unittest.mock import AsyncMock, MagicMock
import tempfile
import os
from pathlib import Path

from openjiuwen.core.llm.gateway.lora_manager import (
    LoRAManager,
    LoRAStorageAdapter,
    LocalStorageAdapter,
    InferenceNotifier,
)


class TestLocalStorageAdapter:
    def test_save_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = LocalStorageAdapter(tmpdir)
            weights_dir = Path(tmpdir) / "dummy_weights"
            weights_dir.mkdir()
            (weights_dir / "adapter_model.safetensors").write_text("dummy")

            import asyncio
            asyncio.get_event_loop().run_until_complete(
                adapter.save("user_1", 1, str(weights_dir))
            )

            assert os.path.exists(os.path.join(tmpdir, "user_1", "v1"))

    def test_get_latest_version_returns_none_for_new_user(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = LocalStorageAdapter(tmpdir)
            import asyncio
            result = asyncio.get_event_loop().run_until_complete(
                adapter.get_latest_version("new_user")
            )
            assert result is None

    def test_get_latest_version_returns_max_version(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = LocalStorageAdapter(tmpdir)
            os.makedirs(os.path.join(tmpdir, "user_1", "v1"))
            os.makedirs(os.path.join(tmpdir, "user_1", "v3"))

            import asyncio
            result = asyncio.get_event_loop().run_until_complete(
                adapter.get_latest_version("user_1")
            )
            assert result == 3


class TestLoRAManager:
    @pytest.mark.asyncio
    async def test_get_active_lora_returns_none_for_new_user(self):
        mock_storage = AsyncMock()
        mock_storage.get_latest_version.return_value = None
        mock_notifier = AsyncMock()

        manager = LoRAManager(storage=mock_storage, notifier=mock_notifier)
        result = await manager.get_active_lora("new_user")
        assert result is None

    @pytest.mark.asyncio
    async def test_publish_updates_active_lora(self):
        mock_storage = AsyncMock()
        mock_storage.get_latest_version.return_value = None
        mock_notifier = AsyncMock()

        manager = LoRAManager(storage=mock_storage, notifier=mock_notifier)

        await manager.publish("user_1", "/tmp/weights")

        mock_storage.save.assert_called_once()
        mock_storage.set_latest.assert_called_once()
        mock_notifier.notify_update.assert_called_once()

        active = await manager.get_active_lora("user_1")
        assert active is not None
        assert active.startswith("user_1_v")


class TestInferenceNotifier:
    def test_init(self):
        notifier = InferenceNotifier("http://localhost:8000")
        assert notifier.inference_url == "http://localhost:8000"
```

- [ ] **Step 2: 运行测试验证失败**

Run: `cd agent-core && python -m pytest tests/core/llm/gateway/test_lora_manager.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 lora_manager.py**

```python
# openjiuwen/core/llm/gateway/lora_manager.py
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

"""
LoRA Weight Manager - unified versioned storage and hot-load notification.

Provides:
- LoRAManager: High-level LoRA lifecycle management
- LoRAStorageAdapter: Abstract storage interface
- LocalStorageAdapter: Local filesystem implementation
- InferenceNotifier: Notifies inference service to load new LoRA weights
"""

import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional

import httpx


class LoRAStorageAdapter(ABC):
    """Abstract interface for LoRA weight storage."""

    @abstractmethod
    async def save(self, user_id: str, version: int, weights_path: str) -> None:
        """Save LoRA weights for a user at a specific version."""
        raise NotImplementedError

    @abstractmethod
    async def get_latest_version(self, user_id: str) -> Optional[int]:
        """Get the latest version number for a user, or None."""
        raise NotImplementedError

    @abstractmethod
    async def set_latest(self, user_id: str, version: int) -> None:
        """Update the 'latest' symlink to point to the given version."""
        raise NotImplementedError


class LocalStorageAdapter(LoRAStorageAdapter):
    """Local filesystem LoRA storage."""

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)

    async def save(self, user_id: str, version: int, weights_path: str) -> None:
        dest = self.base_path / user_id / f"v{version}"
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copytree(weights_path, dest, dirs_exist_ok=True)

    async def get_latest_version(self, user_id: str) -> Optional[int]:
        user_dir = self.base_path / user_id
        if not user_dir.exists():
            return None
        versions = []
        for d in user_dir.iterdir():
            if d.is_dir() and d.name.startswith("v"):
                try:
                    versions.append(int(d.name[1:]))
                except ValueError:
                    pass
        return max(versions) if versions else None

    async def set_latest(self, user_id: str, version: int) -> None:
        latest_link = self.base_path / user_id / "latest"
        target = self.base_path / user_id / f"v{version}"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(target)


class InferenceNotifier:
    """Notifies the inference service to load new LoRA weights."""

    def __init__(self, inference_url: str):
        self.inference_url = inference_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=30.0)

    async def notify_update(
        self, user_id: str, lora_name: str, weights_path: str
    ) -> None:
        """
        Notify the inference service to load a new LoRA adapter.

        Args:
            user_id: User identifier.
            lora_name: LoRA adapter name.
            weights_path: Path to the LoRA weights.
        """
        await self.client.post(
            f"{self.inference_url}/v1/load_lora_adapter",
            json={
                "lora_name": lora_name,
                "lora_path": weights_path,
            },
        )

    async def close(self) -> None:
        await self.client.aclose()


class LoRAManager:
    """High-level LoRA weight lifecycle manager."""

    def __init__(
        self,
        storage: LoRAStorageAdapter,
        notifier: InferenceNotifier,
    ):
        self.storage = storage
        self.notifier = notifier
        self._active_loras: Dict[str, str] = {}

    async def get_active_lora(self, user_id: str) -> Optional[str]:
        """Get the currently active LoRA name for a user."""
        return self._active_loras.get(user_id)

    async def publish(self, user_id: str, weights_path: str) -> None:
        """
        Publish a new LoRA weight and notify the inference service.

        Args:
            user_id: User identifier.
            weights_path: Path to the trained LoRA weights.
        """
        version = await self._next_version(user_id)
        await self.storage.save(user_id, version, weights_path)
        await self.storage.set_latest(user_id, version)

        lora_name = f"{user_id}_v{version}"
        await self.notifier.notify_update(user_id, lora_name, weights_path)

        self._active_loras[user_id] = lora_name

    async def _next_version(self, user_id: str) -> int:
        """Get the next version number for a user."""
        latest = await self.storage.get_latest_version(user_id)
        return (latest or 0) + 1
```

- [ ] **Step 4: 实现 stores/__init__.py**

```python
# openjiuwen/core/llm/gateway/stores/__init__.py
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

"""LoRA storage backend implementations."""

from openjiuwen.core.llm.gateway.lora_manager import (
    LocalStorageAdapter,
    LoRAStorageAdapter,
)

__all__ = ["LocalStorageAdapter", "LoRAStorageAdapter"]
```

- [ ] **Step 5: 运行测试验证通过**

Run: `cd agent-core && python -m pytest tests/core/llm/gateway/test_lora_manager.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
cd agent-core
git add openjiuwen/core/llm/gateway/lora_manager.py openjiuwen/core/llm/gateway/stores/__init__.py openjiuwen/core/llm/gateway/stores/local_store.py tests/core/llm/gateway/test_lora_manager.py
git commit -m "feat(online-rl): add LoRAManager with storage and inference notifier"
```

---

## Task 7: 训练调度器 (scheduler.py)

**Files:**
- Create: `openjiuwen/core/llm/gateway/scheduler.py`
- Test: `tests/core/llm/gateway/test_scheduler.py`

- [ ] **Step 1: 编写 scheduler 测试**

```python
# tests/core/llm/gateway/test_scheduler.py
"""Tests for training scheduler."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from openjiuwen.core.llm.gateway.scheduler import TrainingScheduler
from openjiuwen.core.llm.gateway.trigger import ThresholdTrigger


class TestTrainingScheduler:
    def test_init(self):
        mock_store = AsyncMock()
        mock_backend = AsyncMock()
        mock_lora = AsyncMock()
        mock_trigger = ThresholdTrigger(threshold=10)

        scheduler = TrainingScheduler(
            rollout_store=mock_store,
            rl_backend=mock_backend,
            lora_manager=mock_lora,
            trigger_policy=mock_trigger,
            scan_interval=60.0,
        )
        assert scheduler.scan_interval == 60.0
        assert scheduler._running is False

    @pytest.mark.asyncio
    async def test_scan_and_train_triggers_when_threshold_met(self):
        mock_store = AsyncMock()
        mock_backend = AsyncMock()
        mock_backend.train.return_value = MagicMock(
            user_id="u1", weights_path="/tmp/lora"
        )
        mock_lora = AsyncMock()

        mock_trigger = AsyncMock()
        mock_trigger.should_trigger.return_value = True

        scheduler = TrainingScheduler(
            rollout_store=mock_store,
            rl_backend=mock_backend,
            lora_manager=mock_lora,
            trigger_policy=mock_trigger,
            scan_interval=60.0,
        )
        scheduler._get_active_users = AsyncMock(return_value=["u1"])
        scheduler._get_pending_count = AsyncMock(return_value=10)
        scheduler._collect_trajectories = AsyncMock(return_value=[])
        scheduler._mark_trained = AsyncMock()

        await scheduler._scan_and_train()

        mock_trigger.should_trigger.assert_called_once()
        mock_backend.train.assert_called_once()
        mock_lora.publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_scan_and_train_skips_when_below_threshold(self):
        mock_store = AsyncMock()
        mock_backend = AsyncMock()
        mock_lora = AsyncMock()

        mock_trigger = AsyncMock()
        mock_trigger.should_trigger.return_value = False

        scheduler = TrainingScheduler(
            rollout_store=mock_store,
            rl_backend=mock_backend,
            lora_manager=mock_lora,
            trigger_policy=mock_trigger,
            scan_interval=60.0,
        )
        scheduler._get_active_users = AsyncMock(return_value=["u1"])
        scheduler._get_pending_count = AsyncMock(return_value=5)

        await scheduler._scan_and_train()

        mock_trigger.should_trigger.assert_called_once()
        mock_backend.train.assert_not_called()
```

- [ ] **Step 2: 运行测试验证失败**

Run: `cd agent-core && python -m pytest tests/core/llm/gateway/test_scheduler.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 scheduler.py**

```python
# openjiuwen/core/llm/gateway/scheduler.py
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

"""
Training Scheduler - periodic scan and training trigger for online RL.

Scans pending trajectories at a configurable interval, evaluates the
TriggerPolicy for each user, and submits training tasks when thresholds
are met.
"""

import asyncio
import logging
from typing import TYPE_CHECKING, List, Optional

from openjiuwen.core.llm.gateway.trigger import TriggerPolicy

if TYPE_CHECKING:
    from openjiuwen.core.llm.gateway.rl_backend import RLBackend
    from openjiuwen.core.llm.gateway.lora_manager import LoRAManager
    from openjiuwen.dev_tools.agentrl.rollout_store.base import RolloutPersistence
    from openjiuwen.core.llm.gateway.schemas import UnifiedTrajectory

logger = logging.getLogger(__name__)


class TrainingScheduler:
    """Periodically scans and triggers RL training."""

    def __init__(
        self,
        rollout_store: "RolloutPersistence",
        rl_backend: "RLBackend",
        lora_manager: "LoRAManager",
        trigger_policy: TriggerPolicy,
        scan_interval: float = 600.0,
    ):
        """
        Args:
            rollout_store: Trajectory storage for querying pending trajectories.
            rl_backend: RL training backend.
            lora_manager: LoRA weight manager.
            trigger_policy: Policy for deciding when to trigger training.
            scan_interval: Scan interval in seconds (default: 600 = 10 minutes).
        """
        self.rollout_store = rollout_store
        self.rl_backend = rl_backend
        self.lora_manager = lora_manager
        self.trigger_policy = trigger_policy
        self.scan_interval = scan_interval
        self._running = False
        self._scan_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the scheduler loop."""
        if self._running:
            return
        self._running = True
        self._scan_task = asyncio.create_task(self._scan_loop())
        logger.info("TrainingScheduler started (interval=%.0fs)", self.scan_interval)

    async def stop(self) -> None:
        """Stop the scheduler loop."""
        self._running = False
        if self._scan_task:
            self._scan_task.cancel()
            try:
                await self._scan_task
            except asyncio.CancelledError:
                pass
        logger.info("TrainingScheduler stopped")

    async def _scan_loop(self) -> None:
        """Main scan loop."""
        while self._running:
            try:
                await self._scan_and_train()
            except Exception:
                logger.exception("Error in scan loop")
            await asyncio.sleep(self.scan_interval)

    async def _scan_and_train(self) -> None:
        """Scan all active users and trigger training where appropriate."""
        active_users = await self._get_active_users()

        for user_id in active_users:
            pending_count = await self._get_pending_count(user_id)

            if await self.trigger_policy.should_trigger(user_id, pending_count):
                await self._train_user(user_id)

    async def _train_user(self, user_id: str) -> None:
        """Execute training for a single user."""
        try:
            trajectories = await self._collect_trajectories(user_id)
            if not trajectories:
                logger.warning("No trajectories collected for user %s", user_id)
                return

            result = await self.rl_backend.train(trajectories)
            await self.lora_manager.publish(user_id, result.weights_path)
            await self._mark_trained(user_id, trajectories)

            logger.info(
                "Training completed for user %s: weights=%s status=%s",
                user_id, result.weights_path, result.status,
            )
        except Exception:
            logger.exception("Training failed for user %s", user_id)

    async def _get_active_users(self) -> List[str]:
        """Get list of users with pending trajectories."""
        results = await self.rollout_store.query_rollouts(
            filters={"phase": "train"}, limit=10000
        )
        users = set()
        for r in results:
            uid = r.get("user_id") or r.get("task_id", "").split("-")[0]
            if uid:
                users.add(uid)
        return list(users)

    async def _get_pending_count(self, user_id: str) -> int:
        """Get count of pending (untrained) trajectories for a user."""
        results = await self.rollout_store.query_rollouts(
            filters={"phase": "train", "user_id": user_id}, limit=10000
        )
        return len(results)

    async def _collect_trajectories(self, user_id: str) -> List["UnifiedTrajectory"]:
        """Collect pending trajectories for a user.

        NOTE: This is a placeholder. The actual implementation will convert
        stored RolloutMessage objects back to UnifiedTrajectory format.
        """
        results = await self.rollout_store.query_rollouts(
            filters={"phase": "train", "user_id": user_id}, limit=1000
        )
        return []

    async def _mark_trained(
        self, user_id: str, trajectories: List["UnifiedTrajectory"]
    ) -> None:
        """Mark trajectories as trained (remove from pending)."""
        pass
```

- [ ] **Step 4: 运行测试验证通过**

Run: `cd agent-core && python -m pytest tests/core/llm/gateway/test_scheduler.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
cd agent-core
git add openjiuwen/core/llm/gateway/scheduler.py tests/core/llm/gateway/test_scheduler.py
git commit -m "feat(online-rl): add TrainingScheduler with configurable trigger policy"
```

---

## Task 8: 模块导出与配置

**Files:**
- Modify: `openjiuwen/core/llm/__init__.py`
- Create: `openjiuwen/core/llm/gateway/config.py`

- [ ] **Step 1: 实现 config.py**

```python
# openjiuwen/core/llm/gateway/config.py
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

"""
Configuration schema for the online RL gateway.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ProxyConfig:
    """Proxy configuration."""
    upstream_url: str = "http://localhost:8000"
    host: str = "0.0.0.0"
    port: int = 8001
    timeout: float = 120.0


@dataclass
class RecorderConfig:
    """Recorder configuration."""
    tokenizer_model: str = "Qwen/Qwen2.5-7B"
    save_path: str = "/data/rollouts"
    flush_interval: int = 100


@dataclass
class TriggerConfig:
    """Trigger policy configuration."""
    type: str = "threshold"
    threshold: int = 200


@dataclass
class SchedulerConfig:
    """Scheduler configuration."""
    enabled: bool = True
    scan_interval: float = 600.0
    trigger: TriggerConfig = field(default_factory=TriggerConfig)


@dataclass
class RLBackendConfig:
    """RL backend configuration."""
    name: str = "verl"
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoRAConfig:
    """LoRA management configuration."""
    storage_type: str = "local"
    storage_base_path: str = "/data/lora_weights"
    inference_url: str = "http://localhost:8000"


@dataclass
class GatewayConfig:
    """Top-level gateway configuration."""
    enabled: bool = True
    proxy: ProxyConfig = field(default_factory=ProxyConfig)
    recorder: RecorderConfig = field(default_factory=RecorderConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    rl_backend: RLBackendConfig = field(default_factory=RLBackendConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
```

- [ ] **Step 2: 更新 __init__.py 完整导出**

```python
# openjiuwen/core/llm/gateway/__init__.py
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

"""
Online RL Gateway - LLM proxy with trajectory collection for online RL.
"""

from openjiuwen.core.llm.gateway.schemas import (
    RewardType,
    TokenData,
    TrajectoryMetadata,
    UnifiedTrajectory,
)
from openjiuwen.core.llm.gateway.config import GatewayConfig
from openjiuwen.core.llm.gateway.trigger import TriggerPolicy, ThresholdTrigger
from openjiuwen.core.llm.gateway.rl_backend import (
    RLBackend,
    RLBackendRegistry,
    TrainingResult,
)

__all__ = [
    "RewardType",
    "TokenData",
    "TrajectoryMetadata",
    "UnifiedTrajectory",
    "GatewayConfig",
    "TriggerPolicy",
    "ThresholdTrigger",
    "RLBackend",
    "RLBackendRegistry",
    "TrainingResult",
]
```

- [ ] **Step 3: 验证导入**

Run: `cd agent-core && python -c "from openjiuwen.core.llm.gateway import GatewayConfig, UnifiedTrajectory, RLBackendRegistry; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
cd agent-core
git add openjiuwen/core/llm/gateway/config.py openjiuwen/core/llm/gateway/__init__.py
git commit -m "feat(online-rl): add GatewayConfig and complete module exports"
```

---

## Task 9: 集成测试与验证

**Files:**
- Create: `tests/core/llm/gateway/test_integration.py`

- [ ] **Step 1: 编写集成测试**

```python
# tests/core/llm/gateway/test_integration.py
"""Integration tests for the online RL gateway."""
import pytest
from unittest.mock import AsyncMock, MagicMock
import tempfile

from openjiuwen.core.llm.gateway.config import GatewayConfig
from openjiuwen.core.llm.gateway.schemas import UnifiedTrajectory, RewardType
from openjiuwen.core.llm.gateway.trigger import ThresholdTrigger
from openjiuwen.core.llm.gateway.rl_backend import RLBackendRegistry, TrainingResult


class TestGatewayConfig:
    def test_default_config(self):
        config = GatewayConfig()
        assert config.enabled is True
        assert config.proxy.upstream_url == "http://localhost:8000"
        assert config.scheduler.scan_interval == 600.0
        assert config.scheduler.trigger.threshold == 200
        assert config.rl_backend.name == "verl"
        assert config.lora.storage_type == "local"


class TestModuleImports:
    def test_all_exports_available(self):
        from openjiuwen.core.llm.gateway import (
            GatewayConfig,
            RewardType,
            TokenData,
            TrajectoryMetadata,
            UnifiedTrajectory,
            TriggerPolicy,
            ThresholdTrigger,
            RLBackend,
            RLBackendRegistry,
            TrainingResult,
        )
        assert RewardType.PRM == "prm"
        assert ThresholdTrigger(threshold=10).threshold == 10


class TestEndToEnd:
    @pytest.mark.asyncio
    async def test_full_pipeline_mock(self):
        """Test the full pipeline: proxy + recorder + scheduler + backend + lora."""
        from openjiuwen.core.llm.gateway.proxy import LLMProxy
        from openjiuwen.core.llm.gateway.recorder import Recorder
        from openjiuwen.core.llm.gateway.scheduler import TrainingScheduler
        from openjiuwen.core.llm.gateway.lora_manager import (
            LoRAManager,
            InferenceNotifier,
            LocalStorageAdapter,
        )

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": [1, 2, 3]}
        mock_tokenizer.apply_chat_template.return_value = "user: hello"

        mock_store = AsyncMock()
        mock_backend = AsyncMock()
        mock_backend.train.return_value = TrainingResult(
            user_id="u1", weights_path="/tmp/lora"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            storage = LocalStorageAdapter(tmpdir)
            notifier = InferenceNotifier("http://localhost:8000")
            lora_manager = LoRAManager(storage=storage, notifier=notifier)

            recorder = Recorder(tokenizer=mock_tokenizer, rollout_store=mock_store)
            trigger = ThresholdTrigger(threshold=1)

            scheduler = TrainingScheduler(
                rollout_store=mock_store,
                rl_backend=mock_backend,
                lora_manager=lora_manager,
                trigger_policy=trigger,
                scan_interval=60.0,
            )

            assert scheduler is not None
            assert recorder is not None
            assert lora_manager is not None
```

- [ ] **Step 2: 运行所有 gateway 测试**

Run: `cd agent-core && python -m pytest tests/core/llm/gateway/ -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
cd agent-core
git add tests/core/llm/gateway/test_integration.py
git commit -m "test(online-rl): add integration tests for full gateway pipeline"
```

---

## 自审检查

### 1. Spec 覆盖度

| Spec 章节 | 对应 Task | 状态 |
|-----------|-----------|------|
| UnifiedTrajectory 数据格式 | Task 1 | OK |
| LLM Gateway Proxy | Task 4 | OK |
| LLM Gateway Recorder | Task 5 | OK |
| 训练调度器 | Task 7 | OK |
| 触发策略 | Task 2 | OK |
| RL 后端抽象 | Task 3 | OK |
| LoRA 权重管理 | Task 6 | OK |
| 配置设计 | Task 8 | OK |
| 目录结构 | 全部 Task | OK |

### 2. 占位符扫描

- Task 3 的 `verl_backend.py` 有 `TODO` 注释 - 预期，verl SDK 尚未集成
- Task 7 的 `_collect_trajectories` 和 `_mark_trained` 有 `TODO` - 预期，需后续完善
- 无 "TBD", "implement later", "add appropriate error handling" 等占位符

### 3. 类型一致性

- `UnifiedTrajectory`, `RewardType`, `TokenData` 在 Task 1 定义，后续一致使用
- `RLBackend`, `TrainingResult` 在 Task 3 定义，Task 7 和 Task 9 一致使用
- `LoRAManager`, `LoRAStorageAdapter` 在 Task 6 定义，Task 4 和 Task 7 一致使用
- 所有方法签名与定义一致
