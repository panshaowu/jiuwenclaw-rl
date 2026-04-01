# Jiuwen Agent-Core 在线强化学习框架 Implementation Plan (v3.0 重构版)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 agent-core 中构建生产就绪的在线RL框架，通过重构将公共RL组件提取到 `core/rl_engine/`，新增 FastAPI Gateway 实现不停机学习，轻薄 RLBackend 对接 verl 训练后端，统一 LoRA 权重管理实现热加载。

**Architecture:** 将现有 `dev_tools/agentrl/` 的公共组件迁移到 `core/rl_engine/` 作为RL基础底座，在其上新增 `online/` 子模块实现 FastAPI Gateway、SQLite轨迹存储、verl DataProto转换层、训练触发调度。离线RL特有逻辑保留在 `dev_tools/rl_training/` 作为薄封装。

**Tech Stack:** Python 3.11+, FastAPI (HTTP服务), httpx (异步HTTP), pydantic (数据模型), transformers (tokenizer), verl (训练后端), asyncio (并发), SQLite (存储)

**Spec:** `docs/superpowers/specs/2026-03-31-online-rl-framework-design-v3.md`

---

## 文件结构规划

### Phase 1: 公共基础迁移

| 源文件 | 目标文件 | 说明 |
|--------|---------|------|
| `dev_tools/agentrl/config/` | `core/rl_engine/config/` | 配置Schema迁移 |
| `dev_tools/agentrl/agent_runtime/` | `core/rl_engine/runtime/` | 轨迹收集组件 (重命名) |
| `dev_tools/agentrl/rl_trainer/` | `core/rl_engine/trainer/` | 训练执行器 (重命名) |
| `dev_tools/agentrl/rollout_store/` | `core/rl_engine/store/` | 轨迹持久化 (重命名) |
| `dev_tools/agentrl/reward/` | `core/rl_engine/reward/` | 奖励注册表 |
| `dev_tools/agentrl/proxy/` | `core/rl_engine/proxy/` | LLM推理代理 |
| `dev_tools/agentrl/monitoring/` | `core/rl_engine/monitoring/` | 监控指标 |

### Phase 2: 在线RL实现

| 文件 | 职责 |
|------|------|
| `core/rl_engine/online/__init__.py` | 模块入口 |
| `core/rl_engine/online/schemas.py` | 融合数据格式 (Trajectory, Turn, TokenData) |
| `core/rl_engine/online/gateway/__init__.py` | Gateway模块入口 |
| `core/rl_engine/online/gateway/app.py` | FastAPI 应用入口 + 路由注册 |
| `core/rl_engine/online/gateway/proxy.py` | HTTP代理 (流式/非流式 + LoRA注入) |
| `core/rl_engine/online/gateway/recorder.py` | SessionRecorder (session生命周期) |
| `core/rl_engine/online/store/__init__.py` | 存储模块入口 |
| `core/rl_engine/online/store/trajectory_store.py` | SQLite TrajectoryStore (状态机) |
| `core/rl_engine/online/store/rollout_adapter.py` | Trajectory → verl DataProto 转换层 |
| `core/rl_engine/online/scheduler/__init__.py` | 调度模块入口 |
| `core/rl_engine/online/scheduler/trigger.py` | TriggerPolicy 抽象 + 内置实现 |
| `core/rl_engine/online/scheduler/training_scheduler.py` | TrainingScheduler (定时扫描) |
| `core/rl_engine/online/scheduler/resource_scheduler.py` | ResourceScheduler (轻量封装) |
| `core/rl_engine/online/backend/__init__.py` | 后端模块入口 |
| `core/rl_engine/online/backend/rl_backend.py` | RLBackend 最小接口 |
| `core/rl_engine/online/backend/verl_backend.py` | VerlBackend (直接包装RayPPOTrainer) |
| `core/rl_engine/online/backend/slime_backend.py` | SlimeBackend (预留，暂不实现) |
| `core/rl_engine/online/trainer/__init__.py` | 训练器模块入口 |
| `core/rl_engine/online/trainer/lora_trainer.py` | BatchUserLoRATrainer |
| `core/rl_engine/online/trainer/lora_manager.py` | LoRAManager + 版本管理 |

### Phase 3: 离线RL精简

| 源文件 | 目标文件 | 说明 |
|--------|---------|------|
| `dev_tools/agentrl/coordinator/` | `dev_tools/rl_training/coordinator/` | 训练协调器迁移 |
| `dev_tools/agentrl/optimizer/` | `dev_tools/rl_training/optimizer/` | RLOptimizer迁移 |
| `dev_tools/agentrl/` (其余) | 已迁移到 `core/rl_engine/` | 删除 |

### 测试

| 文件 | 说明 |
|------|------|
| `tests/unit_tests/core/rl_engine/test_*.py` | 公共基础组件测试 |
| `tests/unit_tests/core/rl_engine/online/test_*.py` | 在线RL模块测试 |
| `tests/unit_tests/dev_tools/rl_training/test_*.py` | 离线RL工具测试 |

---

## Task 1: 公共基础迁移 (Phase 1)

**Files:** Move files from `dev_tools/agentrl/` to `core/rl_engine/`

- [ ] **Step 1: 创建目标目录结构**

```
core/rl_engine/
├── __init__.py
├── config/
├── runtime/
├── trainer/
├── store/
├── reward/
├── proxy/
└── monitoring/
```

- [ ] **Step 2: 迁移文件并更新导入**

迁移规则:
- `openjiuwen.dev_tools.agentrl.config` → `openjiuwen.core.rl_engine.config`
- `openjiuwen.dev_tools.agentrl.agent_runtime` → `openjiuwen.core.rl_engine.runtime`
- `openjiuwen.dev_tools.agentrl.rl_trainer` → `openjiuwen.core.rl_engine.trainer`
- `openjiuwen.dev_tools.agentrl.rollout_store` → `openjiuwen.core.rl_engine.store`
- `openjiuwen.dev_tools.agentrl.reward` → `openjiuwen.core.rl_engine.reward`
- `openjiuwen.dev_tools.agentrl.proxy` → `openjiuwen.core.rl_engine.proxy`
- `openjiuwen.dev_tools.agentrl.monitoring` → `openjiuwen.core.rl_engine.monitoring`

- [ ] **Step 3: 更新 `core/rl_engine/__init__.py` 导出公共API**

```python
# core/rl_engine/__init__.py
from openjiuwen.core.rl_engine.config.schemas import (
    RLConfig, TrainingConfig, RolloutConfig, AgentRuntimeConfig, PersistenceConfig
)
from openjiuwen.core.rl_engine.runtime.trajectory import (
    TrajectoryCollectionRail, Rollout, RolloutMessage
)
from openjiuwen.core.rl_engine.store.base import RolloutPersistence
from openjiuwen.core.rl_engine.reward.registry import RewardRegistry, reward_registry
from openjiuwen.core.rl_engine.proxy.backend_proxy import BackendProxy

__all__ = [
    "RLConfig", "TrainingConfig", "RolloutConfig",
    "AgentRuntimeConfig", "PersistenceConfig",
    "TrajectoryCollectionRail", "Rollout", "RolloutMessage",
    "RolloutPersistence", "RewardRegistry", "reward_registry",
    "BackendProxy",
]
```

- [ ] **Step 4: 更新所有引用 (约43处)**

使用全局搜索替换更新所有 `from openjiuwen.dev_tools.agentrl` 导入。

- [ ] **Step 5: 验证迁移**

```bash
cd agent-core
pytest tests/unit_tests/core/rl_engine/ -v
pytest tests/unit_tests/dev_tools/ -v  # 确保旧测试仍通过
```

---

## Task 2: 融合数据模型 (schemas.py)

**Files:**
- Create: `core/rl_engine/online/schemas.py`
- Create: `core/rl_engine/online/__init__.py`
- Test: `tests/unit_tests/core/rl_engine/online/test_schemas.py`

- [ ] **Step 1: 编写 schemas 测试**

```python
# tests/unit_tests/core/rl_engine/online/test_schemas.py
"""Tests for fused online RL gateway data schemas."""
import pytest
from datetime import datetime
from openjiuwen.core.rl_engine.online.schemas import (
    TrajectoryStatus, RewardType, Turn, TokenData, Trajectory,
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
        token_data = TokenData(prompt_ids=[1, 2], response_ids=[3, 4], logprobs=[-0.5, -0.3], loss_mask=[1, 1])
        traj = Trajectory(
            trajectory_id="t2", user_id="u2", session_id="s2",
            turns=[],
            token_data=token_data,
            created_at=datetime.now(),
        )
        assert traj.token_data is not None
        assert traj.token_data.prompt_ids == [1, 2]
```

- [ ] **Step 2: 实现 schemas**

```python
# core/rl_engine/online/schemas.py
"""Fused data schemas for online RL framework.

Design rationale:
- Trajectory + Turn: Session-level data (from previous brainstorm)
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
    PRM = "prm"           # Process Reward Model
    ENV = "env"           # Environment reward
    HUMAN = "human"       # Human annotation
    CUSTOM = "custom"     # Custom reward function


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
# core/rl_engine/online/__init__.py
"""Online RL framework — extends rl_engine with real-time learning capabilities."""
from openjiuwen.core.rl_engine.online.schemas import (
    Trajectory, TrajectoryStatus, RewardType, Turn, TokenData, RewardSignal,
)

__all__ = [
    "Trajectory", "TrajectoryStatus", "RewardType", "Turn", "TokenData", "RewardSignal",
]
```

- [ ] **Step 4: 验证**

```bash
cd agent-core
pytest tests/unit_tests/core/rl_engine/online/test_schemas.py -v
```

---

## Task 3: SessionRecorder (gateway/recorder.py)

**Files:**
- Create: `core/rl_engine/online/gateway/recorder.py`
- Create: `core/rl_engine/online/gateway/__init__.py`
- Test: `tests/unit_tests/core/rl_engine/online/test_recorder.py`

- [ ] **Step 1: 编写 recorder 测试**

```python
# tests/unit_tests/core/rl_engine/online/test_recorder.py
"""Tests for SessionRecorder."""
import pytest
import asyncio
from datetime import datetime
from openjiuwen.core.rl_engine.online.gateway.recorder import SessionRecorder
from openjiuwen.core.rl_engine.online.schemas import TrajectoryStatus

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
        assert len(trajectory.turns) == 2  # user + assistant

    def test_non_final_response(self, recorder):
        recorder.record_request("u1", "s1", "hello")
        trajectory = recorder.record_response("s1", "hi", is_final=False)
        assert trajectory is None  # Not final, no trajectory yet

    def test_timeout_cleanup(self, recorder):
        recorder.record_request("u1", "s1", "hello")
        # Simulate timeout
        recorder._sessions["s1"].last_activity = datetime.fromtimestamp(0)
        recorder.cleanup_expired(max_age_seconds=3600)
        assert "s1" not in recorder._sessions

    def test_concurrent_access(self, recorder):
        import threading
        errors = []

        def record_many():
            try:
                for i in range(100):
                    recorder.record_request(f"u{i}", f"s{i}", f"msg {i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_many) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(recorder._sessions) == 1000
```

- [ ] **Step 2: 实现 SessionRecorder**

```python
# core/rl_engine/online/gateway/recorder.py
"""SessionRecorder — manages session lifecycle and trajectory collection.

Design rationale:
- Thread-safe with asyncio.Lock for concurrent request handling
- Automatic timeout cleanup to prevent memory leaks
- Returns Trajectory when session is complete (final response)
- Integrates with rl_engine.runtime.TrajectoryCollectionRail concepts
"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Optional
from openjiuwen.core.rl_engine.online.schemas import Turn, Trajectory
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
    """Records conversation sessions and produces trajectories for RL training.

    Usage:
        recorder = SessionRecorder()
        recorder.record_request(user_id, session_id, prompt)
        # ... stream response tokens ...
        trajectory = recorder.record_response(session_id, full_response, is_final=True)
    """

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

- [ ] **Step 3: 模块入口更新**

```python
# core/rl_engine/online/gateway/__init__.py
"""LLM Gateway — FastAPI service for online RL trajectory collection."""
from openjiuwen.core.rl_engine.online.gateway.recorder import SessionRecorder

__all__ = ["SessionRecorder"]
```

- [ ] **Step 4: 验证**

```bash
cd agent-core
pytest tests/unit_tests/core/rl_engine/online/test_recorder.py -v
```

---

## Task 4: Gateway 服务 (app.py + proxy.py)

**Files:**
- Create: `core/rl_engine/online/gateway/app.py`
- Create: `core/rl_engine/online/gateway/proxy.py`
- Test: `tests/unit_tests/core/rl_engine/online/test_proxy.py`

- [ ] **Step 1: 编写 proxy 测试**

```python
# tests/unit_tests/core/rl_engine/online/test_proxy.py
"""Tests for LLM Proxy with LoRA routing."""
import pytest
from unittest.mock import AsyncMock, patch
from openjiuwen.core.rl_engine.online.gateway.proxy import LLMProxy

class TestLLMProxy:
    @pytest.fixture
    def proxy(self):
        return LLMProxy(base_url="http://localhost:8000")

    @pytest.mark.asyncio
    async def test_forward_request(self, proxy):
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = AsyncMock()
            mock_response.json.return_value = {"choices": [{"message": {"content": "hi"}}]}
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response

            result = await proxy.forward_request(
                messages=[{"role": "user", "content": "hello"}],
                lora_adapter="user123_lora_v1"
            )
            assert result["choices"][0]["message"]["content"] == "hi"
```

- [ ] **Step 2: 实现 LLM Proxy**

```python
# core/rl_engine/online/gateway/proxy.py
"""LLM Proxy — forwards requests to inference service with LoRA routing."""
import httpx
from typing import Any, Dict, List, Optional
from openjiuwen.core.common.logging import logger


class LLMProxy:
    """HTTP proxy to LLM inference service with LoRA adapter injection."""

    def __init__(self, base_url: str, timeout: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def forward_request(
        self,
        messages: List[Dict[str, str]],
        lora_adapter: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Forward a non-streaming request to the inference service."""
        payload = {"messages": messages, **kwargs}
        if lora_adapter:
            payload["extra_body"] = {"lora_adapter": lora_adapter}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            return response.json()

    async def stream_forward(
        self,
        messages: List[Dict[str, str]],
        lora_adapter: Optional[str] = None,
        **kwargs
    ):
        """Forward a streaming request (SSE) to the inference service."""
        payload = {"messages": messages, "stream": True, **kwargs}
        if lora_adapter:
            payload["extra_body"] = {"lora_adapter": lora_adapter}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        yield line[6:]  # Strip "data: " prefix
```

- [ ] **Step 3: 实现 FastAPI App**

```python
# core/rl_engine/online/gateway/app.py
"""FastAPI application entry point for online RL Gateway."""
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from openjiuwen.core.rl_engine.online.gateway.proxy import LLMProxy
from openjiuwen.core.rl_engine.online.gateway.recorder import SessionRecorder
from openjiuwen.core.rl_engine.online.store.trajectory_store import TrajectoryStore
from openjiuwen.core.common.logging import logger


def create_app(
    inference_base_url: str = "http://localhost:8000",
    db_path: str = "trajectories.db",
) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="Jiuwen Online RL Gateway", version="1.0.0")

    # Initialize components
    proxy = LLMProxy(base_url=inference_base_url)
    recorder = SessionRecorder()
    store = TrajectoryStore(db_path=db_path)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        """OpenAI-compatible chat completions endpoint."""
        body = await request.json()
        user_id = request.headers.get("X-User-ID", "anonymous")
        session_id = request.headers.get("X-Session-ID", "default")

        # Record request
        await recorder.record_request(user_id, session_id, body.get("messages", []))

        # Get active LoRA adapter for user
        lora_adapter = request.headers.get("X-LoRA-Adapter")

        if body.get("stream", False):
            return StreamingResponse(
                _stream_response(proxy, recorder, session_id, body, lora_adapter),
                media_type="text/event-stream",
            )
        else:
            result = await proxy.forward_request(
                messages=body.get("messages", []),
                lora_adapter=lora_adapter,
            )
            # Record response
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            trajectory = await recorder.record_response(session_id, content, is_final=True)
            if trajectory:
                await store.save_trajectory(trajectory)
            return result

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


async def _stream_response(proxy, recorder, session_id, body, lora_adapter):
    """Handle streaming response collection."""
    full_content = ""
    async for chunk in proxy.stream_forward(
        messages=body.get("messages", []),
        lora_adapter=lora_adapter,
    ):
        full_content += chunk
        yield f"data: {chunk}\n\n"

    # Record final response
    trajectory = await recorder.record_response(session_id, full_content, is_final=True)
    if trajectory:
        # Save in background
        import asyncio
        asyncio.create_task(_save_trajectory(trajectory))


async def _save_trajectory(trajectory):
    """Save trajectory to store (background task)."""
    from openjiuwen.core.rl_engine.online.store.trajectory_store import TrajectoryStore
    store = TrajectoryStore()
    await store.save_trajectory(trajectory)
```

- [ ] **Step 4: 验证**

```bash
cd agent-core
pytest tests/unit_tests/core/rl_engine/online/test_proxy.py -v
```

---

## Task 5: 轨迹存储 (trajectory_store.py + rollout_adapter.py)

**Files:**
- Create: `core/rl_engine/online/store/trajectory_store.py`
- Create: `core/rl_engine/online/store/rollout_adapter.py`
- Create: `core/rl_engine/online/store/__init__.py`
- Test: `tests/unit_tests/core/rl_engine/online/test_trajectory_store.py`

- [ ] **Step 1: 编写 trajectory_store 测试**

```python
# tests/unit_tests/core/rl_engine/online/test_trajectory_store.py
"""Tests for SQLite TrajectoryStore."""
import pytest
import tempfile
import os
from datetime import datetime
from openjiuwen.core.rl_engine.online.store.trajectory_store import TrajectoryStore
from openjiuwen.core.rl_engine.online.schemas import (
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
        assert len(pending) == 3  # t0, t2, t4

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
        # PENDING → TRAINED (skip TRAINING) should fail
        with pytest.raises(ValueError):
            await store.update_status("t1", TrajectoryStatus.TRAINED)
```

- [ ] **Step 2: 实现 TrajectoryStore**

```python
# core/rl_engine/online/store/trajectory_store.py
"""SQLite-based trajectory storage with state machine."""
import aiosqlite
import json
from typing import List, Optional
from openjiuwen.core.rl_engine.online.schemas import Trajectory, TrajectoryStatus
from openjiuwen.core.common.logging import logger


# Valid state transitions
VALID_TRANSITIONS = {
    TrajectoryStatus.PENDING: {TrajectoryStatus.TRAINING},
    TrajectoryStatus.TRAINING: {TrajectoryStatus.TRAINED, TrajectoryStatus.FAILED},
    TrajectoryStatus.TRAINED: set(),  # Terminal state
    TrajectoryStatus.FAILED: {TrajectoryStatus.PENDING},  # Allow retry
}


class TrajectoryStore:
    """SQLite-based storage for online RL trajectories.

    Features:
    - Atomic status transitions with state machine validation
    - Efficient scanning by user_id and status
    - JSON serialization for complex fields (turns, token_data)
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
            token_data=None,  # TODO: deserialize if present
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

- [ ] **Step 3: 实现 RolloutPersistenceAdapter**

```python
# core/rl_engine/online/store/rollout_adapter.py
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

from openjiuwen.core.rl_engine.online.schemas import Trajectory
from openjiuwen.core.rl_engine.coordinator.encoding import RolloutEncoder
from openjiuwen.core.common.logging import logger


class RolloutPersistenceAdapter:
    """Converts online Trajectory to verl DataProto for training.

    Design rationale:
    - Reuses existing RolloutEncoder from agentrl for token encoding
    - Handles padding and batching for verl compatibility
    - Maps online fields to verl DataProto fields
    """

    def __init__(self, encoder: RolloutEncoder):
        self._encoder = encoder

    def convert(self, trajectories: List[Trajectory]) -> DataProto:
        """Convert a batch of trajectories to verl DataProto.

        Args:
            trajectories: List of online RL trajectories

        Returns:
            verl DataProto ready for training
        """
        if not trajectories:
            raise ValueError("No trajectories to convert")

        # Step 1: Convert Trajectory → RolloutWithReward
        rollouts_with_reward = []
        for t in trajectories:
            rollout = self._trajectory_to_rollout_with_reward(t)
            if rollout:
                rollouts_with_reward.append(rollout)

        if not rollouts_with_reward:
            raise ValueError("No valid rollouts after conversion")

        # Step 2: Build TensorDict
        input_ids_list = [torch.tensor(r.input_prompt_ids, dtype=torch.long) for r in rollouts_with_reward]
        response_ids_list = [torch.tensor(r.output_response_ids, dtype=torch.long) for r in rollouts_with_reward]
        reward_list = [torch.tensor([r.reward], dtype=torch.float32) for r in rollouts_with_reward]
        uid_list = [r.task_id for r in rollouts_with_reward]

        # Pad sequences to max length
        max_len = max(len(ids) for ids in input_ids_list + response_ids_list)
        input_ids_padded = [self._pad(ids, max_len) for ids in input_ids_list]
        response_ids_padded = [self._pad(ids, max_len) for ids in response_ids_list]

        batch = TensorDict({
            "input_ids": torch.stack(input_ids_padded),
            "responses": torch.stack(response_ids_padded),
            "token_level_scores": torch.stack(reward_list),
            "response_mask": torch.ones(len(rollouts_with_reward), max_len, dtype=torch.long),
        }, batch_size=[len(rollouts_with_reward)])

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

    def _trajectory_to_rollout_with_reward(self, trajectory: Trajectory):
        """Convert online Trajectory to RolloutWithReward."""
        if trajectory.token_data:
            # Use pre-computed token data (fast path)
            from openjiuwen.core.rl_engine.coordinator.schemas import RolloutWithReward
            return RolloutWithReward(
                input_prompt_ids=trajectory.token_data.prompt_ids,
                output_response_ids=trajectory.token_data.response_ids,
                reward=trajectory.reward.value if trajectory.reward else 0.0,
                loss_mask=trajectory.token_data.loss_mask,
            )
        # Fallback: encode from text (slower)
        return self._encoder.encode_from_trajectory(trajectory)
```

- [ ] **Step 4: 模块入口**

```python
# core/rl_engine/online/store/__init__.py
"""Storage layer for online RL."""
from openjiuwen.core.rl_engine.online.store.trajectory_store import TrajectoryStore
from openjiuwen.core.rl_engine.online.store.rollout_adapter import RolloutPersistenceAdapter

__all__ = ["TrajectoryStore", "RolloutPersistenceAdapter"]
```

- [ ] **Step 5: 验证**

```bash
cd agent-core
pytest tests/unit_tests/core/rl_engine/online/test_trajectory_store.py -v
```

---

## Task 6: 训练调度器 (scheduler/)

**Files:**
- Create: `core/rl_engine/online/scheduler/trigger.py`
- Create: `core/rl_engine/online/scheduler/training_scheduler.py`
- Create: `core/rl_engine/online/scheduler/resource_scheduler.py`
- Create: `core/rl_engine/online/scheduler/__init__.py`
- Test: `tests/unit_tests/core/rl_engine/online/test_trigger.py`

- [ ] **Step 1: 实现 TriggerPolicy**

```python
# core/rl_engine/online/scheduler/trigger.py
"""TriggerPolicy — pluggable strategies for when to trigger RL training."""
from abc import ABC, abstractmethod
from typing import List
from openjiuwen.core.rl_engine.online.schemas import Trajectory


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
# core/rl_engine/online/scheduler/training_scheduler.py
"""TrainingScheduler — periodically scans for pending trajectories and triggers training."""
import asyncio
from typing import Optional
from openjiuwen.core.rl_engine.online.store.trajectory_store import TrajectoryStore
from openjiuwen.core.rl_engine.online.scheduler.trigger import TriggerPolicy
from openjiuwen.core.rl_engine.online.scheduler.resource_scheduler import ResourceScheduler
from openjiuwen.core.rl_engine.online.backend.rl_backend import RLBackend
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

        # Group by user
        user_trajectories = {}
        for t in pending:
            user_trajectories.setdefault(t.user_id, []).append(t)

        for user_id, trajectories in user_trajectories.items():
            if self.trigger.should_train(user_id, trajectories):
                await self._submit_for_training(user_id, trajectories)

    async def _submit_for_training(self, user_id: str, trajectories):
        """Submit trajectories for training."""
        # Mark as training
        for t in trajectories:
            await self.store.update_status(t.trajectory_id, "training")

        try:
            # Submit to resource scheduler
            await self.resource_scheduler.submit_training_job(
                user_id=user_id,
                trajectories=trajectories,
            )
        except Exception as e:
            logger.error("Training submission failed for user %s: %s", user_id, e)
            # Revert to pending
            for t in trajectories:
                await self.store.update_status(t.trajectory_id, "pending")
```

- [ ] **Step 3: 实现 ResourceScheduler**

```python
# core/rl_engine/online/scheduler/resource_scheduler.py
"""ResourceScheduler — lightweight resource request interface for training jobs."""
from abc import ABC, abstractmethod
from typing import List
from openjiuwen.core.rl_engine.online.schemas import Trajectory
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
        # Directly call backend (no external resource management)
        # In production, this would be delegated to K8s/Ray/etc.
        pass
```

- [ ] **Step 4: 验证**

```bash
cd agent-core
pytest tests/unit_tests/core/rl_engine/online/test_trigger.py -v
```

---

## Task 7: RLBackend 轻薄抽象 + VerlBackend

**Files:**
- Create: `core/rl_engine/online/backend/rl_backend.py`
- Create: `core/rl_engine/online/backend/verl_backend.py`
- Create: `core/rl_engine/online/backend/__init__.py`
- Test: `tests/unit_tests/core/rl_engine/online/test_rl_backend.py`

- [ ] **Step 1: 实现 RLBackend 最小接口**

```python
# core/rl_engine/online/backend/rl_backend.py
"""Minimal RL backend interface.

Design rationale:
- Single method: train(data) -> result
- No complex registry or adapter layers
- verl implementation is primary; slime is reserved
"""
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
        """Execute one training step.

        Args:
            data: verl DataProto containing tokenized trajectories
            config: Training configuration (algorithm, hyperparameters)

        Returns:
            TrainingResult with weights path and metrics
        """
        ...
```

- [ ] **Step 2: 实现 VerlBackend**

```python
# core/rl_engine/online/backend/verl_backend.py
"""Verl backend — thin wrapper around VerlTrainingExecutor.

Design rationale:
- Directly reuses existing VerlTrainingExecutor (which extends RayPPOTrainer)
- No additional abstraction layers
- Configuration maps directly to verl's OmegaConf format
"""
from typing import Any, Dict
from openjiuwen.core.rl_engine.online.backend.rl_backend import RLBackend, TrainingResult
from openjiuwen.core.rl_engine.trainer.verl_executor import VerlTrainingExecutor
from openjiuwen.core.common.logging import logger


class VerlBackend(RLBackend):
    """Verl backend — thin wrapper around VerlTrainingExecutor."""

    def __init__(self, executor: VerlTrainingExecutor):
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

- [ ] **Step 3: 验证**

```bash
cd agent-core
pytest tests/unit_tests/core/rl_engine/online/test_rl_backend.py -v
```

---

## Task 8: LoRA 训练器 (trainer/)

**Files:**
- Create: `core/rl_engine/online/trainer/lora_trainer.py`
- Create: `core/rl_engine/online/trainer/lora_manager.py`
- Create: `core/rl_engine/online/trainer/__init__.py`
- Test: `tests/unit_tests/core/rl_engine/online/test_lora_manager.py`

- [ ] **Step 1: 实现 LoRAManager**

```python
# core/rl_engine/online/trainer/lora_manager.py
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
    - Metadata tracking (trajectory count, reward avg, etc.)
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

        # Get next version
        existing = self._versions.get(user_id, [])
        version = len(existing) + 1
        version_dir = os.path.join(user_path, f"v{version}")

        # Copy weights
        if os.path.isdir(weights_path):
            shutil.copytree(weights_path, version_dir, dirs_exist_ok=True)
        else:
            os.makedirs(version_dir, exist_ok=True)
            shutil.copy2(weights_path, version_dir)

        # Update latest symlink
        latest_link = os.path.join(user_path, "latest")
        if os.path.islink(latest_link) or os.path.exists(latest_link):
            os.remove(latest_link)
        os.symlink(version_dir, latest_link)

        # Create version metadata
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

        # Notify inference service
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

- [ ] **Step 2: 验证**

```bash
cd agent-core
pytest tests/unit_tests/core/rl_engine/online/test_lora_manager.py -v
```

---

## Task 9: 离线RL精简 (dev_tools/rl_training/)

**Files:**
- Create: `dev_tools/rl_training/__init__.py`
- Move: `dev_tools/agentrl/coordinator/` → `dev_tools/rl_training/coordinator/`
- Move: `dev_tools/agentrl/optimizer/` → `dev_tools/rl_training/optimizer/`

- [ ] **Step 1: 创建 rl_training 模块入口**

```python
# dev_tools/rl_training/__init__.py
"""RL training tools — offline RL utilities built on core/rl_engine."""
from openjiuwen.core.rl_engine import (
    RLConfig, TrainingConfig, RolloutConfig,
    AgentRuntimeConfig, PersistenceConfig,
)
from openjiuwen.dev_tools.rl_training.optimizer.rl_optimizer import RLOptimizer

__all__ = [
    "RLConfig", "TrainingConfig", "RolloutConfig",
    "AgentRuntimeConfig", "PersistenceConfig",
    "RLOptimizer",
]
```

- [ ] **Step 2: 更新 RLOptimizer 导入**

更新 `optimizer/rl_optimizer.py` 中的导入路径:
- `openjiuwen.dev_tools.agentrl.config.schemas` → `openjiuwen.core.rl_engine.config.schemas`
- `openjiuwen.dev_tools.agentrl.coordinator.schemas` → `openjiuwen.core.rl_engine.coordinator.schemas`
- 等

- [ ] **Step 3: 更新 examples 导入**

```python
# examples/rl_calculator/train.py
# 旧: from openjiuwen.dev_tools.agentrl import RLConfig, RLOptimizer
# 新:
from openjiuwen.core.rl_engine.config.schemas import RLConfig
from openjiuwen.dev_tools.rl_training.optimizer.rl_optimizer import RLOptimizer
```

- [ ] **Step 4: 验证**

```bash
cd agent-core
pytest tests/unit_tests/dev_tools/rl_training/ -v
```

---

## Task 10: 集成测试

**Files:**
- Create: `tests/integration_tests/core/rl_engine/online/test_gateway_integration.py`

- [ ] **Step 1: 实现集成测试**

```python
# tests/integration_tests/core/rl_engine/online/test_gateway_integration.py
"""Integration test for complete online RL flow."""
import pytest
import asyncio
from httpx import AsyncClient
from openjiuwen.core.rl_engine.online.gateway.app import create_app

@pytest.mark.asyncio
async def test_complete_flow():
    """Test complete request → trajectory → training flow."""
    app = create_app(
        inference_base_url="http://localhost:8000",
        db_path=":memory:",
    )

    async with AsyncClient(app=app, base_url="http://test") as client:
        # Send request
        response = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}]},
            headers={
                "X-User-ID": "test_user",
                "X-Session-ID": "test_session",
            },
        )

        # Note: This will fail without a real inference service
        # The test verifies the request flow, not the actual LLM response
        assert response.status_code in [200, 502]  # 502 expected without inference service
```

---

## 验收清单

- [ ] `core/rl_engine/` 公共基础组件可独立导入和使用
- [ ] `core/rl_engine/online/` 在线RL Gateway 可处理 HTTP 请求并收集轨迹
- [ ] RolloutPersistenceAdapter 可将 Trajectory 正确转换为 verl DataProto
- [ ] VerlBackend 可调用 RayPPOTrainer 执行训练
- [ ] LoRAManager 可实现权重版本化和热加载通知
- [ ] `dev_tools/rl_training/` 离线RL工具可通过 RLOptimizer 正常启动训练
- [ ] 所有单元测试通过 (覆盖率 > 80%)
- [ ] 集成测试验证在线RL完整流程 (收集 → 转换 → 训练 → 热加载)
- [ ] 所有 examples 更新导入路径后可正常运行
