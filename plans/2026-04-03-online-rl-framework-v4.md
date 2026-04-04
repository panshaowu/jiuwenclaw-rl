# Jiuwen Agent-Core 在线强化学习框架实施计划 (v4.0 - 纯SDK版)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 agent-core 中构建生产就绪的纯SDK模式在线RL框架，将RL功能作为`agent_evolving`的子模块，通过RAIL钩子机制注入，实现与现有自演进能力的协同工作。

**Architecture:** 将现有 `dev_tools/agentrl/` 的公共组件迁移到 `agent_evolving/rl/`，作为智能体自演进能力的一部分。新增 EvolutionOrchestrator 统一管理规则引擎演进和RL演进。离线RL特有逻辑保留在 `dev_tools/rl_training/` 作为薄封装。

**Tech Stack:** Python 3.11+, pydantic (数据模型), asyncio (并发), SQLite (存储), verl (训练后端)

**Spec:** `docs/superpowers/specs/2026-04-03-online-rl-framework-design-v4.md`

---

## 文件结构规划

### Phase 1: 核心组件迁移与创建

| 源文件 | 目标文件 | 说明 |
|--------|---------|------|
| `dev_tools/agentrl/config/` | `agent_evolving/rl/config.py` | 配置Schema迁移并简化 |
| `dev_tools/agentrl/agent_runtime/trajectory.py` | `agent_evolving/rl/trajectory_rail.py` | RAIL钩子迁移，适配AgentRail基类 |
| `dev_tools/agentrl/rollout_store/` | `agent_evolving/rl/store/` | 轨迹持久化迁移 |
| `dev_tools/agentrl/reward/` | `agent_evolving/rl/reward_scorer.py` | 奖励计算迁移 |
| - | `agent_evolving/rl/models.py` | 新增：融合数据模型 |
| - | `agent_evolving/rl/collector.py` | 新增：RolloutCollector核心组件 |
| - | `agent_evolving/rl/gateway/proxy.py` | 新增：LLMAgentProxy进程内代理 |
| - | `agent_evolving/rl/gateway/recorder.py` | 新增：会话轨迹记录器 |

### Phase 2: 存储与转换层

| 文件 | 职责 |
|------|------|
| `agent_evolving/rl/store/__init__.py` | 存储模块入口 |
| `agent_evolving/rl/store/trajectory_store.py` | SQLite TrajectoryStore (状态机) |
| `agent_evolving/rl/store/rollout_adapter.py` | Trajectory → verl DataProto 转换层 |

### Phase 3: 训练与调度

| 文件 | 职责 |
|------|------|
| `agent_evolving/rl/scheduler/__init__.py` | 调度模块入口 |
| `agent_evolving/rl/scheduler/trigger.py` | TriggerPolicy 抽象 + 内置实现 |
| `agent_evolving/rl/scheduler/training_scheduler.py` | TrainingScheduler (定时扫描) |
| `agent_evolving/rl/scheduler/resource_scheduler.py` | ResourceScheduler (轻量封装) |
| `agent_evolving/rl/backend/__init__.py` | 后端模块入口 |
| `agent_evolving/rl/backend/rl_backend.py` | RLBackend 最小接口 |
| `agent_evolving/rl/backend/verl_backend.py` | VerlBackend (直接包装RayPPOTrainer) |
| `agent_evolving/rl/trainer/__init__.py` | 训练器模块入口 |
| `agent_evolving/rl/trainer/lora_trainer.py` | BatchUserLoRATrainer |
| `agent_evolving/rl/trainer/lora_manager.py` | LoRAManager + 版本管理 |

### Phase 4: 协同编排

| 文件 | 职责 |
|------|------|
| `agent_evolving/orchestrator.py` | EvolutionOrchestrator + ResourceCoordinator |

### Phase 5: 离线RL精简

| 源文件 | 目标文件 | 说明 |
|--------|---------|------|
| `dev_tools/agentrl/coordinator/` | `dev_tools/rl_training/coordinator/` | 训练协调器迁移 |
| `dev_tools/agentrl/optimizer/` | `dev_tools/rl_training/optimizer/` | RLOptimizer迁移 |
| `dev_tools/agentrl/` (其余) | 已迁移到 `agent_evolving/rl/` | 删除 |

### 测试

| 文件 | 说明 |
|------|------|
| `tests/unit_tests/agent_evolving/rl/test_*.py` | RL模块测试 |
| `tests/unit_tests/agent_evolving/test_orchestrator.py` | 协同编排测试 |
| `tests/integration_tests/agent_evolving/rl/test_integration.py` | 集成测试 |

---

## Task 1: 数据模型 (models.py)

**Files:**
- Create: `agent_evolving/rl/models.py`
- Create: `agent_evolving/rl/__init__.py`
- Test: `tests/unit_tests/agent_evolving/rl/test_models.py`

- [ ] **Step 1: 编写 models 测试**

```python
# tests/unit_tests/agent_evolving/rl/test_models.py
"""Tests for online RL data models."""
import pytest
from datetime import datetime
from openjiuwen.agent_evolving.rl.models import (
    TrajectoryStatus, RewardType, Turn, TokenData, Trajectory, RewardSignal,
)

class TestTrajectoryStatus:
    def test_all_values(self):
        assert TrajectoryStatus.PENDING == "pending"
        assert TrajectoryStatus.TRAINING == "training"
        assert TrajectoryStatus.TRAINED == "trained"
        assert TrajectoryStatus.FAILED == "failed"

class TestRewardType:
    def test_all_values(self):
        assert RewardType.PRM == "prm"
        assert RewardType.ENV == "env"
        assert RewardType.HUMAN == "human"
        assert RewardType.CUSTOM == "custom"

class TestTurn:
    def test_minimal_turn(self):
        turn = Turn(role="user", content="hello", timestamp=datetime.now())
        assert turn.role == "user"
        assert turn.token_count == 0
        assert turn.tool_calls is None

class TestTokenData:
    def test_token_data_defaults(self):
        td = TokenData(prompt_ids=[1, 2, 3], response_ids=[4, 5])
        assert td.prompt_ids == [1, 2, 3]
        assert td.logprobs == []
        assert td.loss_mask == []
        assert td.multimodal_tokens is None

class TestTrajectory:
    def test_minimal_trajectory(self):
        traj = Trajectory(
            trajectory_id="t1", user_id="u1", session_id="s1",
            turns=[Turn(role="user", content="hi", timestamp=datetime.now())],
            created_at=datetime.now(),
        )
        assert traj.status == TrajectoryStatus.PENDING
        assert traj.reward is None
        assert traj.reward_type == RewardType.CUSTOM
        assert traj.token_data is None

    def test_full_trajectory_with_token_data(self):
        token_data = TokenData(
            prompt_ids=[1, 2], response_ids=[3, 4],
            logprobs=[-0.5, -0.3], loss_mask=[1, 1]
        )
        traj = Trajectory(
            trajectory_id="t2", user_id="u2", session_id="s2",
            turns=[],
            token_data=token_data,
            created_at=datetime.now(),
        )
        assert traj.token_data is not None
        assert traj.token_data.prompt_ids == [1, 2]
```

- [ ] **Step 2: 实现 models**

```python
# agent_evolving/rl/models.py
"""Fused data models for online RL framework.

Design rationale:
- Trajectory + Turn: Session-level data
- TokenData: Token-level data for RL training (avoids re-computation)
- RewardSignal: Flexible reward with type and details
- TrajectoryStatus: Complete state machine for training lifecycle
"""
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class TrajectoryStatus(str, Enum):
    """State machine for trajectory training lifecycle."""
    PENDING = "pending"
    TRAINING = "training"
    TRAINED = "trained"
    FAILED = "failed"


class RewardType(str, Enum):
    """Types of reward signals."""
    PRM = "prm"
    ENV = "env"
    HUMAN = "human"
    CUSTOM = "custom"


class Turn(BaseModel):
    """Single turn in a conversation."""
    role: str
    content: str
    timestamp: datetime
    token_count: int = 0
    tool_calls: Optional[List[Dict[str, Any]]] = None


class TokenData(BaseModel):
    """Token-level data for RL training (optional, computed at record time)."""
    prompt_ids: List[int]
    response_ids: List[int]
    logprobs: List[float] = Field(default_factory=list)
    loss_mask: List[int] = Field(default_factory=list)
    multimodal_tokens: Optional[Any] = None


class RewardSignal(BaseModel):
    """Reward signal for a trajectory."""
    value: float
    reward_type: RewardType = RewardType.CUSTOM
    details: Dict[str, Any] = Field(default_factory=dict)


class Trajectory(BaseModel):
    """Complete session trajectory for online RL.

    This is the primary data structure for online RL. It captures
    the full conversation session and optionally includes pre-computed
    token data for efficient RL training.
    """
    trajectory_id: str
    user_id: str
    session_id: str
    turns: List[Turn]
    status: TrajectoryStatus = TrajectoryStatus.PENDING
    reward: Optional[RewardSignal] = None
    reward_type: RewardType = RewardType.CUSTOM
    token_data: Optional[TokenData] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
```

- [ ] **Step 3: 模块入口**

```python
# agent_evolving/rl/__init__.py
"""Online RL framework — extends agent_evolving with RL-based evolution capabilities."""
from openjiuwen.agent_evolving.rl.models import (
    Trajectory, TrajectoryStatus, RewardType, Turn, TokenData, RewardSignal,
)
from openjiuwen.agent_evolving.rl.config import OnlineRLConfig
from openjiuwen.agent_evolving.rl.collector import RolloutCollector

__all__ = [
    "Trajectory", "TrajectoryStatus", "RewardType", "Turn", "TokenData", "RewardSignal",
    "OnlineRLConfig", "RolloutCollector",
]
```

- [ ] **Step 4: 验证**

```bash
cd agent-core
pytest tests/unit_tests/agent_evolving/rl/test_models.py -v
```

---

## Task 2: 配置类 (config.py)

**Files:**
- Create: `agent_evolving/rl/config.py`
- Test: `tests/unit_tests/agent_evolving/rl/test_config.py`

- [ ] **Step 1: 编写 config 测试**

```python
# tests/unit_tests/agent_evolving/rl/test_config.py
"""Tests for OnlineRLConfig."""
import pytest
from openjiuwen.agent_evolving.rl.config import OnlineRLConfig, SamplingConfig, TrainingConfig

class TestOnlineRLConfig:
    def test_default_config(self):
        config = OnlineRLConfig()
        assert config.enabled is False
        assert config.sampling_enabled is False
        assert config.sampling_rate == 0.1

    def test_from_dict(self):
        config = OnlineRLConfig.from_dict({
            "enabled": True,
            "sampling_rate": 0.2,
            "training": {
                "min_trajectory_count": 20,
            }
        })
        assert config.enabled is True
        assert config.sampling_rate == 0.2
        assert config.training.min_trajectory_count == 20

    def test_validation_sampling_rate(self):
        with pytest.raises(ValueError):
            OnlineRLConfig(sampling_rate=1.5)

    def test_lazy_loading(self):
        config = OnlineRLConfig(enabled=False)
        assert config.should_sample() is False
```

- [ ] **Step 2: 实现 OnlineRLConfig**

```python
# agent_evolving/rl/config.py
"""Configuration for online RL framework.

Design rationale:
- Application-level config: User provides via config file or dict
- SDK provides sensible defaults
- Lazy initialization of heavy components
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, field_validator


class SamplingConfig(BaseModel):
    """Sampling configuration for trajectory collection."""
    enabled: bool = False
    rate: float = Field(default=0.1, ge=0.0, le=1.0)
    max_trajectories_per_user: int = 1000
    exclude_patterns: list[str] = Field(default_factory=list)


class TrainingConfig(BaseModel):
    """Training trigger configuration."""
    enabled: bool = True
    min_trajectory_count: int = 10
    min_interval_seconds: int = 3600
    max_batch_size: int = 100
    algorithm: str = "ppo"


class StorageConfig(BaseModel):
    """Storage configuration."""
    db_path: str = "trajectories.db"
    max_storage_mb: int = 1024
    cleanup_after_training: bool = True


class BackendConfig(BaseModel):
    """RL backend configuration."""
    type: str = "verl"
    checkpoint_dir: str = "checkpoints"
    lora_base_path: str = "lora_weights"


class OnlineRLConfig(BaseModel):
    """Top-level configuration for online RL.

    Usage:
        # From dict (typically from app config)
        config = OnlineRLConfig.from_dict(app_config.get("online_rl", {}))

        # Direct instantiation
        config = OnlineRLConfig(enabled=True, sampling_rate=0.2)
    """
    enabled: bool = False
    sampling_enabled: bool = False
    sampling_rate: float = Field(default=0.1, ge=0.0, le=1.0)

    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    backend: BackendConfig = Field(default_factory=BackendConfig)

    @field_validator("sampling_rate")
    @classmethod
    def validate_sampling_rate(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"sampling_rate must be between 0 and 1, got {v}")
        return v

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OnlineRLConfig":
        """Create config from dictionary."""
        return cls.model_validate(data)

    def should_sample(self) -> bool:
        """Check if sampling should be performed."""
        return self.enabled and self.sampling_enabled and self.sampling_rate > 0

    def should_train(self, trajectory_count: int) -> bool:
        """Check if training should be triggered."""
        return (
            self.enabled
            and self.training.enabled
            and trajectory_count >= self.training.min_trajectory_count
        )
```

- [ ] **Step 3: 验证**

```bash
cd agent-core
pytest tests/unit_tests/agent_evolving/rl/test_config.py -v
```

---

## Task 3: TrajectoryCollectionRail钩子 (trajectory_rail.py)

**Files:**
- Create: `agent_evolving/rl/trajectory_rail.py`
- Test: `tests/unit_tests/agent_evolving/rl/test_trajectory_rail.py`

> **注意**：现有`dev_tools/agentrl/agent_runtime/trajectory.py`已实现`TrajectoryCollectionRail`。
> 本Task将其迁移到`agent_evolving/rl/trajectory_rail.py`并适配`AgentRail`基类。

- [ ] **Step 1: 编写 trajectory_rail 测试**

```python
# tests/unit_tests/agent_evolving/rl/test_trajectory_rail.py
"""Tests for TrajectoryCollectionRail."""
import pytest
from unittest.mock import AsyncMock, MagicMock
from openjiuwen.agent_evolving.rl.trajectory_rail import TrajectoryCollectionRail
from openjiuwen.agent_evolving.rl.config import OnlineRLConfig

class TestTrajectoryCollectionRail:
    @pytest.fixture
    def config(self):
        return OnlineRLConfig(enabled=True, sampling_enabled=True, sampling_rate=1.0)

    @pytest.fixture
    def gateway(self, config):
        gateway = MagicMock()
        gateway._config = config
        gateway._record_input = MagicMock()
        gateway._record_output = MagicMock()
        return gateway

    @pytest.fixture
    def rail(self, gateway):
        return TrajectoryCollectionRail(gateway)

    def test_priority(self, rail):
        assert rail.priority == 100

    @pytest.mark.asyncio
    async def test_before_model_call_sampling_disabled(self, gateway):
        gateway._config.sampling_enabled = False
        rail = TrajectoryCollectionRail(gateway)
        ctx = MagicMock()
        await rail.before_model_call(ctx)
        gateway._record_input.assert_not_called()

    @pytest.mark.asyncio
    async def test_before_model_call_sampling_enabled(self, rail, gateway):
        ctx = MagicMock()
        ctx.inputs = MagicMock()
        ctx.inputs.messages = [{"role": "user", "content": "hello"}]
        ctx.inputs.tools = None

        await rail.before_model_call(ctx)
        gateway._record_input.assert_called_once()
```

- [ ] **Step 2: 实现 TrajectoryCollectionRail**

```python
# agent_evolving/rl/trajectory_rail.py
"""TrajectoryCollectionRail — RAIL hook for trajectory collection.

Design rationale:
- Inherits from AgentRail to integrate with existing hook mechanism
- High priority to ensure execution before other rails
- Fast-path check for sampling_enabled to minimize overhead
- Migrated from dev_tools/agentrl/agent_runtime/trajectory.py
"""
from openjiuwen.core.single_agent.rail.base import AgentRail, AgentCallbackContext
from openjiuwen.core.common.logging import logger


class TrajectoryCollectionRail(AgentRail):
    """RAIL hook for collecting LLM interaction trajectories.

    This hook intercepts LLM calls and records input/output data
    for online RL training. It integrates with the RolloutCollector
    through the LLMAgentProxy.

    Usage:
        config = OnlineRLConfig.from_dict(app_config.get("online_rl", {}))
        collector = RolloutCollector(config)
        rail = collector.get_trajectory_rail()
        if rail:
            agent.add_rail(rail)
    """

    priority = 100

    def __init__(self, gateway):
        self._gateway = gateway

    async def before_model_call(self, ctx: AgentCallbackContext) -> None:
        """LLM call hook - record input data."""
        if not self._gateway._config.sampling_enabled:
            return

        inputs = ctx.inputs
        messages = getattr(inputs, "messages", [])
        tools = getattr(inputs, "tools", None)

        self._gateway._record_input(messages, tools)

    async def after_model_call(self, ctx: AgentCallbackContext) -> None:
        """LLM call hook - record output data."""
        if not self._gateway._config.sampling_enabled:
            return

        response = getattr(ctx.inputs, "response", None)
        self._gateway._record_output(response)

    async def after_tool_call(self, ctx: AgentCallbackContext) -> None:
        """Tool call hook - record tool interactions."""
        if not self._gateway._config.sampling_enabled:
            return

        tool_name = getattr(ctx.inputs, "tool_name", None)
        tool_args = getattr(ctx.inputs, "tool_args", None)
        tool_result = getattr(ctx.inputs, "tool_result", None)

        self._gateway._record_tool_call(tool_name, tool_args, tool_result)
```

- [ ] **Step 3: 验证**

```bash
cd agent-core
pytest tests/unit_tests/agent_evolving/rl/test_trajectory_rail.py -v
```

---

## Task 4: LLMAgentProxy进程内代理 (gateway/proxy.py)

**Files:**
- Create: `agent_evolving/rl/gateway/__init__.py`
- Create: `agent_evolving/rl/gateway/proxy.py`
- Create: `agent_evolving/rl/gateway/recorder.py`
- Test: `tests/unit_tests/agent_evolving/rl/test_gateway.py`

- [ ] **Step 1: 编写 recorder 测试**

```python
# tests/unit_tests/agent_evolving/rl/test_gateway.py
"""Tests for SessionRecorder and LLMAgentProxy."""
import pytest
from datetime import datetime
from openjiuwen.agent_evolving.rl.gateway.recorder import SessionRecorder
from openjiuwen.agent_evolving.rl.models import TrajectoryStatus

class TestSessionRecorder:
    @pytest.fixture
    def recorder(self):
        return SessionRecorder()

    def test_record_request(self, recorder):
        recorder.record_request("u1", "s1", "hello")
        assert "s1" in recorder._sessions
        assert len(recorder._sessions["s1"].turns) == 1

    def test_record_response_creates_trajectory(self, recorder):
        recorder.record_request("u1", "s1", "hello")
        trajectory = recorder.record_response("s1", "hi there", is_final=True)
        assert trajectory is not None
        assert len(trajectory.turns) == 2

    def test_non_final_response(self, recorder):
        recorder.record_request("u1", "s1", "hello")
        trajectory = recorder.record_response("s1", "hi", is_final=False)
        assert trajectory is None

    def test_timeout_cleanup(self, recorder):
        recorder.record_request("u1", "s1", "hello")
        recorder._sessions["s1"].last_activity = datetime.fromtimestamp(0)
        recorder.cleanup_expired(max_age_seconds=3600)
        assert "s1" not in recorder._sessions
```

- [ ] **Step 2: 实现 SessionRecorder**

```python
# agent_evolving/rl/gateway/recorder.py
"""SessionRecorder — manages session lifecycle and trajectory collection.

Design rationale:
- Thread-safe with asyncio.Lock for concurrent request handling
- Automatic timeout cleanup to prevent memory leaks
- Returns Trajectory when session is complete (final response)
"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Optional
from openjiuwen.agent_evolving.rl.models import Turn, Trajectory
from openjiuwen.core.common.logging import logger


class Session:
    """Represents an active conversation session."""

    def __init__(self, user_id: str, session_id: str):
        self.user_id = user_id
        self.session_id = session_id
        self.turns: list[Turn] = []
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self._lock = asyncio.Lock()

    def add_turn(self, role: str, content: str) -> Turn:
        turn = Turn(role=role, content=content, timestamp=datetime.now())
        self.turns.append(turn)
        self.last_activity = datetime.now()
        return turn


class SessionRecorder:
    """Records conversation sessions and produces trajectories for RL training."""

    def __init__(self, max_age_seconds: int = 3600):
        self._sessions: Dict[str, Session] = {}
        self._max_age_seconds = max_age_seconds
        self._lock = asyncio.Lock()

    async def record_request(self, user_id: str, session_id: str, prompt: str) -> Session:
        """Record a user request and create/update session."""
        async with self._lock:
            if session_id not in self._sessions:
                session = Session(user_id, session_id)
                self._sessions[session_id] = session
            else:
                session = self._sessions[session_id]

        async with session._lock:
            session.add_turn("user", prompt)

        return session

    async def record_response(
        self, session_id: str, content: str, is_final: bool = False
    ) -> Optional[Trajectory]:
        """Record an assistant response. Returns Trajectory if session is complete."""
        async with self._lock:
            session = self._sessions.get(session_id)

        if not session:
            logger.warning("Session not found: %s", session_id)
            return None

        async with session._lock:
            session.add_turn("assistant", content)

            if is_final:
                trajectory = self._build_trajectory(session)
                del self._sessions[session_id]
                return trajectory

        return None

    def _build_trajectory(self, session: Session) -> Trajectory:
        """Build a Trajectory from a completed session."""
        import uuid
        return Trajectory(
            trajectory_id=str(uuid.uuid4()),
            user_id=session.user_id,
            session_id=session.session_id,
            turns=session.turns,
            created_at=session.created_at,
        )

    def cleanup_expired(self, max_age_seconds: Optional[int] = None) -> int:
        """Remove expired sessions. Returns count of removed sessions."""
        max_age = max_age_seconds or self._max_age_seconds
        cutoff = datetime.now() - timedelta(seconds=max_age)
        expired = [
            sid for sid, s in self._sessions.items()
            if s.last_activity < cutoff
        ]
        for sid in expired:
            del self._sessions[sid]
        if expired:
            logger.info("Cleaned up %d expired sessions", len(expired))
        return len(expired)
```

- [ ] **Step 3: 实现 LLMAgentProxy**

```python
# agent_evolving/rl/gateway/proxy.py
"""LLMAgentProxy — in-process proxy for trajectory collection.

Design rationale:
- NOT an HTTP proxy (unlike V3)
- Direct integration with RAIL hooks
- Records LLM interactions for RL training
"""
from typing import Any, Dict, List, Optional
from openjiuwen.agent_evolving.rl.config import OnlineRLConfig
from openjiuwen.agent_evolving.rl.gateway.recorder import SessionRecorder
from openjiuwen.agent_evolving.rl.models import Trajectory
from openjiuwen.core.common.logging import logger


class LLMAgentProxy:
    """In-process proxy for collecting LLM interaction trajectories.

    This is the core component that bridges the RAIL hook system
    with the trajectory collection and storage system.

    Usage:
        proxy = LLMAgentProxy(config)
        proxy.start_session(user_id, session_id)
        # ... RAIL hooks call _record_input/_record_output ...
        trajectory = proxy.end_session(session_id)
    """

    def __init__(self, config: OnlineRLConfig):
        self._config = config
        self._recorder = SessionRecorder()
        self._current_session_id: Optional[str] = None
        self._pending_trajectories: List[Trajectory] = []

    def _record_input(self, messages: List[Dict], tools: Optional[Any] = None):
        """Record LLM input (called by TrajectoryCollectionRail)."""
        if not self._config.sampling_enabled:
            return

        logger.debug("Recording input: %d messages", len(messages))
        self._current_messages = messages
        self._current_tools = tools

    def _record_output(self, response: Any):
        """Record LLM output (called by TrajectoryCollectionRail)."""
        if not self._config.sampling_enabled:
            return

        logger.debug("Recording output")
        self._current_response = response

    def _record_tool_call(self, tool_name: str, tool_args: Any, tool_result: Any):
        """Record tool call (called by TrajectoryCollectionRail)."""
        if not self._config.sampling_enabled:
            return

        logger.debug("Recording tool call: %s", tool_name)

    async def start_session(self, user_id: str, session_id: str, prompt: str):
        """Start a new session for trajectory collection."""
        await self._recorder.record_request(user_id, session_id, prompt)
        self._current_session_id = session_id

    async def end_session(self, session_id: str, final_response: str) -> Optional[Trajectory]:
        """End a session and return the trajectory."""
        trajectory = await self._recorder.record_response(
            session_id, final_response, is_final=True
        )
        if trajectory:
            self._pending_trajectories.append(trajectory)
        return trajectory

    def get_pending_trajectories(self) -> List[Trajectory]:
        """Get all pending trajectories."""
        return self._pending_trajectories

    def clear_pending_trajectories(self):
        """Clear pending trajectories after they've been processed."""
        self._pending_trajectories.clear()
```

- [ ] **Step 4: 模块入口**

```python
# agent_evolving/rl/gateway/__init__.py
"""Gateway module — in-process trajectory collection components."""
from openjiuwen.agent_evolving.rl.gateway.proxy import LLMAgentProxy
from openjiuwen.agent_evolving.rl.gateway.recorder import SessionRecorder

__all__ = ["LLMAgentProxy", "SessionRecorder"]
```

- [ ] **Step 5: 验证**

```bash
cd agent-core
pytest tests/unit_tests/agent_evolving/rl/test_gateway.py -v
```

---

## Task 5: RolloutCollector核心组件 (collector.py)

**Files:**
- Create: `agent_evolving/rl/collector.py`
- Test: `tests/unit_tests/agent_evolving/rl/test_collector.py`

- [ ] **Step 1: 编写 collector 测试**

```python
# tests/unit_tests/agent_evolving/rl/test_collector.py
"""Tests for RolloutCollector."""
import pytest
from openjiuwen.agent_evolving.rl.collector import RolloutCollector
from openjiuwen.agent_evolving.rl.config import OnlineRLConfig

class TestRolloutCollector:
    @pytest.fixture
    def config(self):
        return OnlineRLConfig(enabled=True, sampling_enabled=True)

    @pytest.fixture
    def collector(self, config):
        return RolloutCollector(config)

    def test_get_trajectory_rail_disabled(self):
        config = OnlineRLConfig(enabled=False)
        collector = RolloutCollector(config)
        rail = collector.get_trajectory_rail()
        assert rail is None

    def test_get_trajectory_rail_enabled(self, collector):
        rail = collector.get_trajectory_rail()
        assert rail is not None
        assert rail.priority == 100

    @pytest.mark.asyncio
    async def test_collect_trajectory(self, collector):
        await collector.start_session("u1", "s1", "hello")
        trajectory = await collector.end_session("s1", "hi there")
        assert trajectory is not None
        assert trajectory.user_id == "u1"
        assert len(trajectory.turns) == 2
```

- [ ] **Step 2: 实现 RolloutCollector**

```python
# agent_evolving/rl/collector.py
"""RolloutCollector — main entry point for online RL trajectory collection.

Design rationale:
- Single entry point for application developers
- Manages LLMAgentProxy and TrajectoryCollectionRail
- Provides lazy initialization of heavy components
"""
from typing import Optional
from openjiuwen.agent_evolving.rl.config import OnlineRLConfig
from openjiuwen.agent_evolving.rl.gateway.proxy import LLMAgentProxy
from openjiuwen.agent_evolving.rl.trajectory_rail import TrajectoryCollectionRail
from openjiuwen.agent_evolving.rl.models import Trajectory
from openjiuwen.core.common.logging import logger


class RolloutCollector:
    """Main entry point for online RL trajectory collection.

    Usage:
        # Initialize with config
        config = OnlineRLConfig.from_dict(app_config.get("online_rl", {}))
        collector = RolloutCollector(config)

        # Add RAIL hook to agent (only if sampling enabled)
        rail = collector.get_trajectory_rail()
        if rail:
            agent.add_rail(rail)

        # Start/end sessions (typically done by application layer)
        await collector.start_session(user_id, session_id, prompt)
        trajectory = await collector.end_session(session_id, response)
    """

    def __init__(self, config: OnlineRLConfig):
        self._config = config
        self._proxy: Optional[LLMAgentProxy] = None
        self._rail: Optional[TrajectoryCollectionRail] = None
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy initialization of heavy components."""
        if self._initialized:
            return

        if self._config.enabled:
            self._proxy = LLMAgentProxy(self._config)
            logger.info("RolloutCollector initialized (sampling_enabled=%s)",
                       self._config.sampling_enabled)

        self._initialized = True

    def get_trajectory_rail(self) -> Optional[TrajectoryCollectionRail]:
        """Get the TrajectoryCollectionRail hook for agent integration.

        Returns None if sampling is disabled, allowing for fast-path
        optimization in the application layer.
        """
        self._ensure_initialized()

        if not self._config.should_sample():
            return None

        if self._rail is None and self._proxy:
            self._rail = TrajectoryCollectionRail(self._proxy)

        return self._rail

    async def start_session(self, user_id: str, session_id: str, prompt: str):
        """Start a new trajectory collection session."""
        self._ensure_initialized()

        if self._proxy:
            await self._proxy.start_session(user_id, session_id, prompt)

    async def end_session(self, session_id: str, final_response: str) -> Optional[Trajectory]:
        """End a session and return the collected trajectory."""
        if self._proxy:
            return await self._proxy.end_session(session_id, final_response)
        return None

    def get_pending_trajectories(self) -> list[Trajectory]:
        """Get all pending trajectories for training."""
        if self._proxy:
            return self._proxy.get_pending_trajectories()
        return []

    def clear_pending_trajectories(self):
        """Clear pending trajectories after training."""
        if self._proxy:
            self._proxy.clear_pending_trajectories()
```

- [ ] **Step 3: 验证**

```bash
cd agent-core
pytest tests/unit_tests/agent_evolving/rl/test_collector.py -v
```

---

## Task 6: 轨迹存储 (store/trajectory_store.py)

**Files:**
- Create: `agent_evolving/rl/store/__init__.py`
- Create: `agent_evolving/rl/store/trajectory_store.py`
- Test: `tests/unit_tests/agent_evolving/rl/test_trajectory_store.py`

- [ ] **Step 1: 编写 trajectory_store 测试**

```python
# tests/unit_tests/agent_evolving/rl/test_trajectory_store.py
"""Tests for SQLite TrajectoryStore."""
import pytest
import tempfile
import os
from datetime import datetime
from openjiuwen.agent_evolving.rl.store.trajectory_store import TrajectoryStore
from openjiuwen.agent_evolving.rl.models import (
    Trajectory, TrajectoryStatus, Turn, RewardSignal, RewardType
)

class TestTrajectoryStore:
    @pytest.fixture
    def store(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        s = TrajectoryStore(db_path=path)
        yield s
        s.close()
        os.unlink(path)

    @pytest.mark.asyncio
    async def test_save_and_get(self, store):
        traj = Trajectory(
            trajectory_id="t1", user_id="u1", session_id="s1",
            turns=[Turn(role="user", content="hi", timestamp=datetime.now())],
        )
        await store.save_trajectory(traj)
        loaded = await store.get_trajectory("t1")
        assert loaded is not None
        assert loaded.trajectory_id == "t1"
        assert loaded.status == TrajectoryStatus.PENDING

    @pytest.mark.asyncio
    async def test_scan_pending(self, store):
        for i in range(5):
            traj = Trajectory(
                trajectory_id=f"t{i}", user_id=f"u{i % 2}", session_id=f"s{i}",
                turns=[],
            )
            await store.save_trajectory(traj)

        pending = await store.scan_pending(user_id="u0")
        assert len(pending) == 3

    @pytest.mark.asyncio
    async def test_status_transitions(self, store):
        traj = Trajectory(
            trajectory_id="t1", user_id="u1", session_id="s1", turns=[],
        )
        await store.save_trajectory(traj)

        await store.update_status("t1", TrajectoryStatus.TRAINING)
        status = await store.get_status("t1")
        assert status == TrajectoryStatus.TRAINING

        await store.update_status("t1", TrajectoryStatus.TRAINED)
        status = await store.get_status("t1")
        assert status == TrajectoryStatus.TRAINED

    @pytest.mark.asyncio
    async def test_invalid_transition(self, store):
        traj = Trajectory(
            trajectory_id="t1", user_id="u1", session_id="s1", turns=[],
        )
        await store.save_trajectory(traj)
        with pytest.raises(ValueError):
            await store.update_status("t1", TrajectoryStatus.TRAINED)
```

- [ ] **Step 2: 实现 TrajectoryStore**

```python
# agent_evolving/rl/store/trajectory_store.py
"""SQLite-based trajectory storage with state machine."""
import aiosqlite
import json
from typing import List, Optional
from openjiuwen.agent_evolving.rl.models import Trajectory, TrajectoryStatus, Turn, RewardSignal, RewardType
from openjiuwen.core.common.logging import logger


VALID_TRANSITIONS = {
    TrajectoryStatus.PENDING: {TrajectoryStatus.TRAINING},
    TrajectoryStatus.TRAINING: {TrajectoryStatus.TRAINED, TrajectoryStatus.FAILED},
    TrajectoryStatus.TRAINED: set(),
    TrajectoryStatus.FAILED: {TrajectoryStatus.PENDING},
}


class TrajectoryStore:
    """SQLite-based storage for online RL trajectories.

    Features:
    - Atomic status transitions with state machine validation
    - Efficient scanning by user_id and status
    - JSON serialization for complex fields
    """

    def __init__(self, db_path: str = "trajectories.db"):
        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None

    async def _get_db(self) -> aiosqlite.Connection:
        if self._db is None:
            self._db = await aiosqlite.connect(self.db_path)
            self._db.row_factory = aiosqlite.Row
            await self._init_db()
        return self._db

    async def _init_db(self):
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS trajectories (
                trajectory_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                turns_json TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                reward_json TEXT,
                reward_type TEXT DEFAULT 'custom',
                token_data_json TEXT,
                metadata_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT
            )
        """)
        await self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_status
            ON trajectories(user_id, status)
        """)
        await self._db.commit()

    async def save_trajectory(self, trajectory: Trajectory):
        """Save a trajectory to the database."""
        db = await self._get_db()
        await db.execute("""
            INSERT OR REPLACE INTO trajectories
            (trajectory_id, user_id, session_id, turns_json, status,
             reward_json, reward_type, token_data_json, metadata_json,
             created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trajectory.trajectory_id,
            trajectory.user_id,
            trajectory.session_id,
            json.dumps([t.model_dump() for t in trajectory.turns]),
            trajectory.status.value,
            json.dumps(trajectory.reward.model_dump()) if trajectory.reward else None,
            trajectory.reward_type.value,
            json.dumps(trajectory.token_data.model_dump()) if trajectory.token_data else None,
            json.dumps(trajectory.metadata),
            trajectory.created_at.isoformat(),
            trajectory.updated_at.isoformat() if trajectory.updated_at else None,
        ))
        await db.commit()

    async def get_trajectory(self, trajectory_id: str) -> Optional[Trajectory]:
        """Load a trajectory by ID."""
        db = await self._get_db()
        row = await db.execute(
            "SELECT * FROM trajectories WHERE trajectory_id = ?",
            (trajectory_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_trajectory(row)

    async def scan_pending(self, user_id: Optional[str] = None) -> List[Trajectory]:
        """Scan for pending trajectories, optionally filtered by user_id."""
        db = await self._get_db()
        if user_id:
            rows = await db.execute(
                "SELECT * FROM trajectories WHERE user_id = ? AND status = 'pending' ORDER BY created_at",
                (user_id,)
            ).fetchall()
        else:
            rows = await db.execute(
                "SELECT * FROM trajectories WHERE status = 'pending' ORDER BY created_at"
            ).fetchall()
        return [self._row_to_trajectory(r) for r in rows]

    async def update_status(self, trajectory_id: str, new_status: TrajectoryStatus):
        """Update trajectory status with state machine validation."""
        current = await self.get_status(trajectory_id)
        if new_status not in VALID_TRANSITIONS.get(current, set()):
            raise ValueError(f"Invalid transition: {current.value} → {new_status.value}")

        db = await self._get_db()
        from datetime import datetime
        await db.execute("""
            UPDATE trajectories SET status = ?, updated_at = ?
            WHERE trajectory_id = ?
        """, (new_status.value, datetime.now().isoformat(), trajectory_id))
        await db.commit()

    async def get_status(self, trajectory_id: str) -> TrajectoryStatus:
        """Get current status of a trajectory."""
        db = await self._get_db()
        row = await db.execute(
            "SELECT status FROM trajectories WHERE trajectory_id = ?",
            (trajectory_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Trajectory not found: {trajectory_id}")
        return TrajectoryStatus(row[0])

    def _row_to_trajectory(self, row) -> Trajectory:
        """Convert database row to Trajectory object."""
        from datetime import datetime
        return Trajectory(
            trajectory_id=row["trajectory_id"],
            user_id=row["user_id"],
            session_id=row["session_id"],
            turns=[Turn(**t) for t in json.loads(row["turns_json"])],
            status=TrajectoryStatus(row["status"]),
            reward=RewardSignal(**json.loads(row["reward_json"])) if row["reward_json"] else None,
            reward_type=RewardType(row["reward_type"]),
            token_data=None,
            metadata=json.loads(row["metadata_json"]) if row["metadata_json"] else {},
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None,
        )

    def close(self):
        """Close database connection."""
        if self._db:
            import asyncio
            try:
                asyncio.get_event_loop().run_until_complete(self._db.close())
            except RuntimeError:
                pass
```

- [ ] **Step 3: 模块入口**

```python
# agent_evolving/rl/store/__init__.py
"""Storage layer for online RL."""
from openjiuwen.agent_evolving.rl.store.trajectory_store import TrajectoryStore

__all__ = ["TrajectoryStore"]
```

- [ ] **Step 4: 验证**

```bash
cd agent-core
pytest tests/unit_tests/agent_evolving/rl/test_trajectory_store.py -v
```

---

## Task 7: DataProto转换层 (store/rollout_adapter.py)

**Files:**
- Create: `agent_evolving/rl/store/rollout_adapter.py`
- Test: `tests/unit_tests/agent_evolving/rl/test_rollout_adapter.py`

- [ ] **Step 1: 实现 RolloutPersistenceAdapter**

```python
# agent_evolving/rl/store/rollout_adapter.py
"""Converts online Trajectory to verl DataProto.

This is the critical bridge between online RL data collection
and offline RL training infrastructure.

Conversion flow:
    Trajectory (online) → Rollout → RolloutWithReward → DataProto (verl)
"""
from typing import List
import torch
from tensordict import TensorDict
from verl import DataProto

from openjiuwen.agent_evolving.rl.models import Trajectory
from openjiuwen.core.common.logging import logger


class RolloutPersistenceAdapter:
    """Converts online Trajectory to verl DataProto for training.

    Design rationale:
    - Handles padding and batching for verl compatibility
    - Maps online fields to verl DataProto fields
    - Supports pre-computed token data (fast path)
    """

    def __init__(self, tokenizer=None):
        self._tokenizer = tokenizer

    def convert(self, trajectories: List[Trajectory]) -> DataProto:
        """Convert a batch of trajectories to verl DataProto.

        Args:
            trajectories: List of online RL trajectories

        Returns:
            verl DataProto ready for training
        """
        if not trajectories:
            raise ValueError("No trajectories to convert")

        rollouts = []
        for t in trajectories:
            rollout = self._trajectory_to_rollout(t)
            if rollout:
                rollouts.append(rollout)

        if not rollouts:
            raise ValueError("No valid rollouts after conversion")

        input_ids_list = [torch.tensor(r["input_ids"], dtype=torch.long) for r in rollouts]
        response_ids_list = [torch.tensor(r["response_ids"], dtype=torch.long) for r in rollouts]
        reward_list = [torch.tensor([r["reward"]], dtype=torch.float32) for r in rollouts]
        uid_list = [r["trajectory_id"] for r in rollouts]

        max_len = max(len(ids) for ids in input_ids_list + response_ids_list)
        input_ids_padded = [self._pad(ids, max_len) for ids in input_ids_list]
        response_ids_padded = [self._pad(ids, max_len) for ids in response_ids_list]

        batch = TensorDict({
            "input_ids": torch.stack(input_ids_padded),
            "responses": torch.stack(response_ids_padded),
            "token_level_scores": torch.stack(reward_list),
            "response_mask": torch.ones(len(rollouts), max_len, dtype=torch.long),
        }, batch_size=[len(rollouts)])

        non_tensor_batch = {
            "uid": uid_list,
        }

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    def _pad(self, tensor: torch.Tensor, max_len: int) -> torch.Tensor:
        """Pad tensor to max_len with zeros."""
        pad_size = max_len - tensor.size(0)
        if pad_size > 0:
            return torch.nn.functional.pad(tensor, (0, pad_size))
        return tensor

    def _trajectory_to_rollout(self, trajectory: Trajectory):
        """Convert online Trajectory to rollout dict."""
        if trajectory.token_data:
            return {
                "input_ids": trajectory.token_data.prompt_ids,
                "response_ids": trajectory.token_data.response_ids,
                "reward": trajectory.reward.value if trajectory.reward else 0.0,
                "trajectory_id": trajectory.trajectory_id,
            }

        if self._tokenizer:
            text = " ".join([t.content for t in trajectory.turns])
            tokens = self._tokenizer.encode(text)
            return {
                "input_ids": tokens[:len(tokens)//2],
                "response_ids": tokens[len(tokens)//2:],
                "reward": trajectory.reward.value if trajectory.reward else 0.0,
                "trajectory_id": trajectory.trajectory_id,
            }

        return None
```

- [ ] **Step 2: 更新模块入口**

```python
# agent_evolving/rl/store/__init__.py
"""Storage layer for online RL."""
from openjiuwen.agent_evolving.rl.store.trajectory_store import TrajectoryStore
from openjiuwen.agent_evolving.rl.store.rollout_adapter import RolloutPersistenceAdapter

__all__ = ["TrajectoryStore", "RolloutPersistenceAdapter"]
```

---

## Task 8: 训练调度器 (scheduler/)

**Files:**
- Create: `agent_evolving/rl/scheduler/__init__.py`
- Create: `agent_evolving/rl/scheduler/trigger.py`
- Create: `agent_evolving/rl/scheduler/training_scheduler.py`
- Create: `agent_evolving/rl/scheduler/resource_scheduler.py`
- Test: `tests/unit_tests/agent_evolving/rl/test_scheduler.py`

- [ ] **Step 1: 实现 TriggerPolicy**

```python
# agent_evolving/rl/scheduler/trigger.py
"""TriggerPolicy — pluggable strategies for when to trigger RL training."""
from abc import ABC, abstractmethod
from typing import List
from openjiuwen.agent_evolving.rl.models import Trajectory


class TriggerPolicy(ABC):
    """Abstract base for training trigger policies."""

    @abstractmethod
    def should_train(self, user_id: str, pending_trajectories: List[Trajectory]) -> bool:
        """Determine if training should be triggered for a user."""
        ...


class CountBasedTrigger(TriggerPolicy):
    """Trigger when pending trajectory count reaches threshold."""

    def __init__(self, min_count: int = 10):
        self.min_count = min_count

    def should_train(self, user_id: str, pending_trajectories: List[Trajectory]) -> bool:
        return len(pending_trajectories) >= self.min_count


class TimeBasedTrigger(TriggerPolicy):
    """Trigger when time since last training exceeds threshold."""

    def __init__(self, min_interval_seconds: int = 3600):
        self.min_interval_seconds = min_interval_seconds
        self._last_training_time = {}

    def should_train(self, user_id: str, pending_trajectories: List[Trajectory]) -> bool:
        import time
        now = time.time()
        last = self._last_training_time.get(user_id, 0)
        if now - last >= self.min_interval_seconds and pending_trajectories:
            self._last_training_time[user_id] = now
            return True
        return False


class CompositeTrigger(TriggerPolicy):
    """Combine multiple trigger policies (AND logic)."""

    def __init__(self, policies: List[TriggerPolicy]):
        self.policies = policies

    def should_train(self, user_id: str, pending_trajectories: List[Trajectory]) -> bool:
        return all(p.should_train(user_id, pending_trajectories) for p in self.policies)
```

- [ ] **Step 2: 实现 TrainingScheduler**

```python
# agent_evolving/rl/scheduler/training_scheduler.py
"""TrainingScheduler — periodically scans for pending trajectories and triggers training."""
import asyncio
from typing import Optional
from openjiuwen.agent_evolving.rl.store.trajectory_store import TrajectoryStore
from openjiuwen.agent_evolving.rl.scheduler.trigger import TriggerPolicy
from openjiuwen.agent_evolving.rl.scheduler.resource_scheduler import ResourceScheduler
from openjiuwen.agent_evolving.rl.backend.rl_backend import RLBackend
from openjiuwen.core.common.logging import logger


class TrainingScheduler:
    """Periodically scans for pending trajectories and triggers training.

    Usage:
        scheduler = TrainingScheduler(store, trigger, resource_scheduler, backend)
        asyncio.create_task(scheduler.start())
    """

    def __init__(
        self,
        store: TrajectoryStore,
        trigger: TriggerPolicy,
        resource_scheduler: ResourceScheduler,
        backend: RLBackend,
        scan_interval: int = 60,
    ):
        self.store = store
        self.trigger = trigger
        self.resource_scheduler = resource_scheduler
        self.backend = backend
        self.scan_interval = scan_interval
        self._running = False

    async def start(self):
        """Start the scanning loop."""
        self._running = True
        logger.info("TrainingScheduler started (interval: %ds)", self.scan_interval)
        while self._running:
            await self._scan_and_submit()
            await asyncio.sleep(self.scan_interval)

    async def stop(self):
        """Stop the scanning loop."""
        self._running = False
        logger.info("TrainingScheduler stopped")

    async def _scan_and_submit(self):
        """Scan for pending trajectories and submit for training."""
        pending = await self.store.scan_pending()
        if not pending:
            return

        user_trajectories = {}
        for t in pending:
            user_trajectories.setdefault(t.user_id, []).append(t)

        for user_id, trajectories in user_trajectories.items():
            if self.trigger.should_train(user_id, trajectories):
                await self._submit_for_training(user_id, trajectories)

    async def _submit_for_training(self, user_id: str, trajectories):
        """Submit trajectories for training."""
        for t in trajectories:
            await self.store.update_status(t.trajectory_id, "training")

        try:
            await self.resource_scheduler.submit_training_job(
                user_id=user_id,
                trajectories=trajectories,
            )
        except Exception as e:
            logger.error("Training submission failed for user %s: %s", user_id, e)
            for t in trajectories:
                await self.store.update_status(t.trajectory_id, "pending")
```

- [ ] **Step 3: 实现 ResourceScheduler**

```python
# agent_evolving/rl/scheduler/resource_scheduler.py
"""ResourceScheduler — lightweight resource request interface for training jobs."""
from abc import ABC, abstractmethod
from typing import List
from openjiuwen.agent_evolving.rl.models import Trajectory
from openjiuwen.core.common.logging import logger


class ResourceScheduler(ABC):
    """Abstract interface for submitting training jobs with resource requirements."""

    @abstractmethod
    async def submit_training_job(self, user_id: str, trajectories: List[Trajectory]):
        """Submit a training job with resource allocation."""
        ...


class LocalResourceScheduler(ResourceScheduler):
    """Local resource scheduler for development/testing.

    Runs training in-process without external resource management.
    """

    def __init__(self, backend):
        self.backend = backend

    async def submit_training_job(self, user_id: str, trajectories: List[Trajectory]):
        logger.info("Submitting training job for user %s (%d trajectories)", user_id, len(trajectories))
```

- [ ] **Step 4: 模块入口**

```python
# agent_evolving/rl/scheduler/__init__.py
"""Scheduler module for training triggers."""
from openjiuwen.agent_evolving.rl.scheduler.trigger import (
    TriggerPolicy, CountBasedTrigger, TimeBasedTrigger, CompositeTrigger,
)
from openjiuwen.agent_evolving.rl.scheduler.training_scheduler import TrainingScheduler
from openjiuwen.agent_evolving.rl.scheduler.resource_scheduler import ResourceScheduler, LocalResourceScheduler

__all__ = [
    "TriggerPolicy", "CountBasedTrigger", "TimeBasedTrigger", "CompositeTrigger",
    "TrainingScheduler", "ResourceScheduler", "LocalResourceScheduler",
]
```

---

## Task 9: RL后端与LoRA管理 (backend/ + trainer/)

**Files:**
- Create: `agent_evolving/rl/backend/__init__.py`
- Create: `agent_evolving/rl/backend/rl_backend.py`
- Create: `agent_evolving/rl/backend/verl_backend.py`
- Create: `agent_evolving/rl/trainer/__init__.py`
- Create: `agent_evolving/rl/trainer/lora_trainer.py`
- Create: `agent_evolving/rl/trainer/lora_manager.py`

- [ ] **Step 1: 实现 RLBackend 最小接口**

```python
# agent_evolving/rl/backend/rl_backend.py
"""Minimal RL backend interface."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class TrainingResult:
    """Training result from RL backend."""
    weights_path: str
    metrics: Dict[str, float]
    user_id: str
    version: int


class RLBackend(ABC):
    """Minimal RL backend interface."""

    @abstractmethod
    async def train(self, data, config: Dict[str, Any]) -> TrainingResult:
        """Execute one training step."""
        ...
```

- [ ] **Step 2: 实现 VerlBackend**

```python
# agent_evolving/rl/backend/verl_backend.py
"""Verl backend — thin wrapper around VerlTrainingExecutor."""
from typing import Any, Dict
from openjiuwen.agent_evolving.rl.backend.rl_backend import RLBackend, TrainingResult
from openjiuwen.core.common.logging import logger


class VerlBackend(RLBackend):
    """Verl backend — thin wrapper around VerlTrainingExecutor."""

    def __init__(self, executor):
        self._executor = executor

    async def train(self, data, config: Dict[str, Any]) -> TrainingResult:
        """Execute training using VerlTrainingExecutor."""
        logger.info("Starting Verl training step")
        metrics = await self._executor.train_step(data)
        weights_path = self._executor.get_latest_checkpoint()

        return TrainingResult(
            weights_path=weights_path,
            metrics=metrics,
            user_id=data.non_tensor_batch.get("uid", ["unknown"])[0],
            version=self._executor.get_current_version(),
        )
```

- [ ] **Step 3: 实现 LoRAManager**

```python
# agent_evolving/rl/trainer/lora_manager.py
"""LoRAManager — manages LoRA weight versions and hot-loading notifications."""
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from openjiuwen.core.common.logging import logger


@dataclass
class LoRAVersion:
    """Metadata for a LoRA weight version."""
    version: int
    user_id: str
    weights_path: str
    created_at: datetime
    trajectory_count: int
    reward_avg: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)


class LoRAManager:
    """Manages LoRA weight versions for online RL.

    Features:
    - Versioned storage (v1/, v2/, ...)
    - Symlink for latest version
    - Hot-loading notification to inference service
    - Metadata tracking
    """

    def __init__(self, base_path: str, inference_url: str = "http://localhost:8000"):
        self.base_path = base_path
        self.inference_url = inference_url
        self._versions: Dict[str, List[LoRAVersion]] = {}

    async def publish(
        self,
        user_id: str,
        weights_path: str,
        metrics: Dict[str, float],
        trajectory_count: int = 0,
    ) -> LoRAVersion:
        """Publish a new LoRA weight version."""
        user_path = os.path.join(self.base_path, user_id)
        os.makedirs(user_path, exist_ok=True)

        existing = self._versions.get(user_id, [])
        version = len(existing) + 1
        version_dir = os.path.join(user_path, f"v{version}")

        if os.path.isdir(weights_path):
            shutil.copytree(weights_path, version_dir, dirs_exist_ok=True)
        else:
            os.makedirs(version_dir, exist_ok=True)
            shutil.copy2(weights_path, version_dir)

        latest_link = os.path.join(user_path, "latest")
        if os.path.islink(latest_link) or os.path.exists(latest_link):
            os.remove(latest_link)
        os.symlink(version_dir, latest_link)

        lora_version = LoRAVersion(
            version=version,
            user_id=user_id,
            weights_path=version_dir,
            created_at=datetime.now(),
            trajectory_count=trajectory_count,
            reward_avg=metrics.get("reward_avg", 0.0),
            metrics=metrics,
        )

        self._versions.setdefault(user_id, []).append(lora_version)

        await self._notify_inference(user_id, version_dir)

        logger.info("Published LoRA v%d for user %s", version, user_id)
        return lora_version

    async def _notify_inference(self, user_id: str, weights_path: str):
        """Notify inference service to load new LoRA adapter."""
        import httpx
        adapter_name = f"{user_id}_lora_v{self._get_latest_version(user_id)}"
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{self.inference_url}/v1/load_lora_adapter",
                    json={
                        "lora_name": adapter_name,
                        "lora_path": weights_path,
                    }
                )
        except Exception as e:
            logger.warning("Failed to notify inference service: %s", e)

    def _get_latest_version(self, user_id: str) -> int:
        versions = self._versions.get(user_id, [])
        return versions[-1].version if versions else 0

    def get_latest_path(self, user_id: str) -> Optional[str]:
        """Get the path to the latest LoRA weights for a user."""
        latest_link = os.path.join(self.base_path, user_id, "latest")
        if os.path.islink(latest_link):
            return os.readlink(latest_link)
        return None
```

---

## Task 10: EvolutionOrchestrator协同编排 (orchestrator.py)

**Files:**
- Create: `agent_evolving/orchestrator.py`
- Test: `tests/unit_tests/agent_evolving/test_orchestrator.py`

- [ ] **Step 1: 实现 EvolutionOrchestrator**

```python
# agent_evolving/orchestrator.py
"""EvolutionOrchestrator — unified management of evolution strategies."""
from enum import Enum
from typing import Optional, List
from dataclasses import dataclass

from openjiuwen.agent_evolving.online.signal_detector import SignalDetector


class EvolutionStrategy(str, Enum):
    """Evolution strategies."""
    RULE_BASED = "rule_based"
    RL_BASED = "rl_based"
    HYBRID = "hybrid"


@dataclass
class EvolutionDecision:
    """Evolution decision."""
    strategy: EvolutionStrategy
    reason: str
    priority: int
    resource_requirements: dict
    estimated_duration: float


class EvolutionOrchestrator:
    """Evolution strategy orchestrator.

    Coordinates rule-based evolution (prompt/tool/memory) and RL-based evolution.
    """

    def __init__(self, config):
        self.config = config
        self.rule_engine = SignalDetector()
        self.rl_collector = None
        self._evolution_history = []

    def analyze_and_decide(self, messages: List[dict], context: dict) -> EvolutionDecision:
        """Analyze scenario and decide evolution strategy."""

        signals = self.rule_engine.detect(messages)
        rl_ready = self._check_rl_conditions()
        resource_status = self._check_resources()

        if signals and not rl_ready:
            return EvolutionDecision(
                strategy=EvolutionStrategy.RULE_BASED,
                reason="Detected evolution signals, RL conditions not met",
                priority=7,
                resource_requirements={"cpu": 1, "memory": "1GB"},
                estimated_duration=5.0
            )
        elif not signals and rl_ready:
            return EvolutionDecision(
                strategy=EvolutionStrategy.RL_BASED,
                reason="RL training conditions met",
                priority=5,
                resource_requirements={"gpu": 1, "memory": "16GB"},
                estimated_duration=300.0
            )
        elif signals and rl_ready:
            if self._is_critical_signal(signals):
                return EvolutionDecision(
                    strategy=EvolutionStrategy.RULE_BASED,
                    reason="Detected critical evolution signal",
                    priority=9,
                    resource_requirements={"cpu": 1, "memory": "1GB"},
                    estimated_duration=5.0
                )
            else:
                return EvolutionDecision(
                    strategy=EvolutionStrategy.RL_BASED,
                    reason="RL training conditions met, signal non-critical",
                    priority=6,
                    resource_requirements={"gpu": 1, "memory": "16GB"},
                    estimated_duration=300.0
                )
        else:
            return EvolutionDecision(
                strategy=EvolutionStrategy.HYBRID,
                reason="No evolution needed",
                priority=0,
                resource_requirements={},
                estimated_duration=0.0
            )

    def execute_evolution(self, decision: EvolutionDecision):
        """Execute evolution."""
        if decision.strategy == EvolutionStrategy.RULE_BASED:
            pass
        elif decision.strategy == EvolutionStrategy.RL_BASED:
            pass
        self._evolution_history.append(decision)

    def _check_rl_conditions(self) -> bool:
        if self.rl_collector is None:
            return False
        return len(self.rl_collector.get_pending_trajectories()) > 0

    def _check_resources(self) -> dict:
        return {"available": True}

    def _is_critical_signal(self, signals) -> bool:
        return False


class ResourceCoordinator:
    """Resource coordinator for evolution tasks."""

    def __init__(self):
        self._available_resources = {"cpu": 8, "memory": "32GB", "gpu": 1}
        self._allocated_resources = {}

    def can_execute(self, decision: EvolutionDecision) -> bool:
        required = decision.resource_requirements
        for resource, amount in required.items():
            available = self._available_resources.get(resource, 0)
            allocated = self._allocated_resources.get(resource, 0)
            if available - allocated < amount:
                return False
        return True

    def allocate(self, decision: EvolutionDecision):
        for resource, amount in decision.resource_requirements.items():
            self._allocated_resources[resource] = \
                self._allocated_resources.get(resource, 0) + amount

    def release(self, decision: EvolutionDecision):
        for resource, amount in decision.resource_requirements.items():
            self._allocated_resources[resource] = \
                self._allocated_resources.get(resource, 0) - amount
```

---

## Task 11: 离线RL精简 (dev_tools/rl_training/)

**Files:**
- Create: `dev_tools/rl_training/__init__.py`
- Move: `dev_tools/agentrl/coordinator/` → `dev_tools/rl_training/coordinator/`
- Move: `dev_tools/agentrl/optimizer/` → `dev_tools/rl_training/optimizer/`

- [ ] **Step 1: 创建 rl_training 模块入口**

```python
# dev_tools/rl_training/__init__.py
"""RL training tools — offline RL utilities built on agent_evolving.rl."""
from openjiuwen.agent_evolving.rl import (
    OnlineRLConfig, RolloutCollector, Trajectory, TrajectoryStatus,
)
from openjiuwen.dev_tools.rl_training.optimizer.rl_optimizer import RLOptimizer

__all__ = [
    "OnlineRLConfig", "RolloutCollector", "Trajectory", "TrajectoryStatus",
    "RLOptimizer",
]
```

- [ ] **Step 2: 更新 RLOptimizer 导入**

更新 `optimizer/rl_optimizer.py` 中的导入路径:
- `openjiuwen.dev_tools.agentrl.config.schemas` → `openjiuwen.agent_evolving.rl.config`
- `openjiuwen.dev_tools.agentrl.coordinator.schemas` → `openjiuwen.agent_evolving.rl.models`

- [ ] **Step 3: 更新 examples 导入**

```python
# examples/rl_calculator/train.py
from openjiuwen.agent_evolving.rl.config import OnlineRLConfig
from openjiuwen.dev_tools.rl_training.optimizer.rl_optimizer import RLOptimizer
```

---

## Task 12: 集成测试

**Files:**
- Create: `tests/integration_tests/agent_evolving/rl/test_integration.py`

- [ ] **Step 1: 实现集成测试**

```python
# tests/integration_tests/agent_evolving/rl/test_integration.py
"""Integration test for complete online RL flow."""
import pytest
import asyncio
from openjiuwen.agent_evolving.rl import RolloutCollector, OnlineRLConfig

@pytest.mark.asyncio
async def test_complete_flow():
    """Test complete request → trajectory → training flow."""
    config = OnlineRLConfig(
        enabled=True,
        sampling_enabled=True,
        sampling_rate=1.0,
    )
    collector = RolloutCollector(config)

    rail = collector.get_trajectory_rail()
    assert rail is not None

    await collector.start_session("test_user", "test_session", "hello")
    trajectory = await collector.end_session("test_session", "hi there")

    assert trajectory is not None
    assert trajectory.user_id == "test_user"
    assert len(trajectory.turns) == 2

    pending = collector.get_pending_trajectories()
    assert len(pending) == 1
```

---

## 验收清单

- [ ] `agent_evolving/rl/` 模块可独立导入和使用
- [ ] `RolloutCollector` 可正确创建并返回 `TrajectoryCollectionRail`
- [ ] `TrajectoryCollectionRail` 继承自 `AgentRail` 并正确实现钩子方法
- [ ] `TrajectoryStore` 可正确存储和检索轨迹
- [ ] `RolloutPersistenceAdapter` 可将 `Trajectory` 正确转换为 verl `DataProto`
- [ ] `VerlBackend` 可调用训练后端执行训练
- [ ] `LoRAManager` 可实现权重版本化和热加载通知
- [ ] `EvolutionOrchestrator` 可正确协调规则引擎和RL演进
- [ ] `dev_tools/rl_training/` 离线RL工具可通过 `RLOptimizer` 正常启动训练
- [ ] 所有单元测试通过 (覆盖率 > 80%)
- [ ] 集成测试验证在线RL完整流程 (收集 → 转换 → 训练 → 热加载)
- [ ] 所有 examples 更新导入路径后可正常运行
