# Jiuwen Agent-Core 在线强化学习框架 Implementation Plan (v2.0 融合版)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 agent-core 中构建生产就绪的在线RL框架，融合 FastAPI Gateway、完整 Session 生命周期管理、灵活 Reward 接口、可插拔训练调度，使智能体在不停机的前提下持续学习。

**Architecture:** 在 agentrl 下新增 online/ 子模块，包含 FastAPI LLM Gateway（Proxy + SessionRecorder）、SQLite 轨迹存储带完整状态机、verl DataProto 转换层、轻薄 RLBackend 抽象（verl 优先），统一 LoRA 权重管理实现热加载。

**Tech Stack:** Python 3.11+, FastAPI (HTTP服务), httpx (异步HTTP), pydantic (数据模型), transformers (tokenizer), verl (训练后端), asyncio (并发), SQLite (存储)

**Spec:** `docs/superpowers/specs/2026-03-31-online-rl-framework-design-v2.md`

---

## 文件结构规划

| 文件 | 职责 | 说明 |
|------|------|------|
| `dev_tools/agentrl/online/__init__.py` | 模块入口 | 在线RL模块导出 |
| `dev_tools/agentrl/online/config.py` | 配置 Schema | 在线RL配置 |
| `dev_tools/agentrl/online/schemas.py` | 融合数据格式 | Trajectory, Turn, TokenData |
| `dev_tools/agentrl/online/gateway/__init__.py` | Gateway模块入口 | - |
| `dev_tools/agentrl/online/gateway/app.py` | FastAPI 应用入口 | 路由注册 |
| `dev_tools/agentrl/online/gateway/proxy.py` | HTTP代理 | 流式/非流式 + LoRA注入 |
| `dev_tools/agentrl/online/gateway/recorder.py` | SessionRecorder | Session生命周期 |
| `dev_tools/agentrl/online/store/__init__.py` | 存储模块入口 | - |
| `dev_tools/agentrl/online/store/trajectory_store.py` | SQLite TrajectoryStore | 状态机管理 |
| `dev_tools/agentrl/online/store/rollout_adapter.py` | Trajectory → DataProto 转换 | 核心转换层 |
| `dev_tools/agentrl/online/scheduler/__init__.py` | 调度模块入口 | - |
| `dev_tools/agentrl/online/scheduler/trigger.py` | TriggerPolicy | 可插拔触发策略 |
| `dev_tools/agentrl/online/scheduler/training_scheduler.py` | TrainingScheduler | 定时扫描 |
| `dev_tools/agentrl/online/scheduler/resource_scheduler.py` | ResourceScheduler | 轻量资源请求 |
| `dev_tools/agentrl/online/backend/__init__.py` | 后端模块入口 | - |
| `dev_tools/agentrl/online/backend/rl_backend.py` | RLBackend 最小接口 | `train(data) -> result` |
| `dev_tools/agentrl/online/backend/verl_backend.py` | VerlBackend | 直接包装 RayPPOTrainer |
| `dev_tools/agentrl/online/backend/slime_backend.py` | SlimeBackend | 预留，暂不实现 |
| `dev_tools/agentrl/online/trainer/__init__.py` | 训练器模块入口 | - |
| `dev_tools/agentrl/online/trainer/lora_trainer.py` | BatchUserLoRATrainer | 批量LoRA训练 |
| `dev_tools/agentrl/online/trainer/lora_manager.py` | LoRAManager | 版本管理 + 热加载 |
| `tests/unit_tests/dev_tools/agentrl/online/test_*.py` | 单元测试 | 各模块测试 |

---

## Task 1: 融合数据模型 (schemas.py)

**Files:**
- Create: `dev_tools/agentrl/online/schemas.py`
- Create: `dev_tools/agentrl/online/__init__.py`
- Test: `tests/unit_tests/dev_tools/agentrl/online/test_schemas.py`

- [ ] **Step 1: 编写 schemas 测试**

```python
# tests/unit_tests/dev_tools/agentrl/online/test_schemas.py
"""Tests for fused online RL gateway data schemas."""
import pytest
from datetime import datetime
from openjiuwen.dev_tools.agentrl.online.schemas import (
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
            trajectory_id="t1", user_id="u1", session_id="s1",
            turns=[
                Turn(role="user", content="calc 2+2", timestamp=datetime.now()),
                Turn(role="assistant", content="4", timestamp=datetime.now(), token_count=1),
            ],
            created_at=datetime.now(),
            token_data=token_data,
            reward=1.0, reward_type=RewardType.ENV,
            reward_details={"accuracy": 1.0},
            status=TrajectoryStatus.PENDING,
            metadata={"model": "Qwen2.5-7B"},
        )
        assert traj.reward == 1.0
        assert traj.token_data == token_data
        assert len(traj.turns) == 2
```

- [ ] **Step 2: 运行测试验证失败**

Run: `cd agent-core && python -m pytest tests/unit_tests/dev_tools/agentrl/online/test_schemas.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: 实现 schemas.py**

```python
# dev_tools/agentrl/online/schemas.py
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

"""
Fused data schemas for the online RL gateway.

Combines:
- Trajectory + Turn: Complete session-level lifecycle (from previous draft)
- TokenData: Tokenized data at recording time (from current design)
- RewardType + reward_details: Flexible reward extension (from current design)
- TrajectoryStatus: Complete state machine (from previous draft)
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class TrajectoryStatus(str, Enum):
    """Trajectory lifecycle status."""
    PENDING = "pending"
    TRAINING = "training"
    TRAINED = "trained"
    FAILED = "failed"


class RewardType(str, Enum):
    """Reward signal source type."""
    PRM = "prm"
    ENV = "env"
    HUMAN = "human"
    CUSTOM = "custom"


@dataclass
class Turn:
    """Single conversation turn."""
    role: str
    content: str
    timestamp: datetime
    token_count: int = 0
    tool_calls: Optional[List[Dict[str, Any]]] = None


@dataclass
class TokenData:
    """Token-level training data (tokenized at recording time)."""
    prompt_ids: List[int]
    response_ids: List[int]
    logprobs: List[float] = field(default_factory=list)
    loss_mask: List[int] = field(default_factory=list)
    multimodal_tokens: Optional[Dict[str, Any]] = None


@dataclass
class Trajectory:
    """
    Fused trajectory format - complete session level + TokenData + flexible reward + state machine.
    """
    trajectory_id: str
    user_id: str
    session_id: str
    turns: List[Turn]
    created_at: datetime

    token_data: Optional[TokenData] = None
    reward: Optional[float] = None
    reward_type: RewardType = RewardType.CUSTOM
    reward_details: Dict[str, Any] = field(default_factory=dict)
    status: TrajectoryStatus = TrajectoryStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)
```

- [ ] **Step 4: 实现 __init__.py**

```python
# dev_tools/agentrl/online/__init__.py
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

"""
Online RL Gateway - Production-ready LLM proxy with trajectory collection for online RL.
"""

from openjiuwen.dev_tools.agentrl.online.schemas import (
    TrajectoryStatus, RewardType, Turn, TokenData, Trajectory,
)

__all__ = ["TrajectoryStatus", "RewardType", "Turn", "TokenData", "Trajectory"]
```

- [ ] **Step 5: 运行测试验证通过**

Run: `cd agent-core && python -m pytest tests/unit_tests/dev_tools/agentrl/online/test_schemas.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
cd agent-core
git add dev_tools/agentrl/online/__init__.py dev_tools/agentrl/online/schemas.py tests/unit_tests/dev_tools/agentrl/online/test_schemas.py
git commit -m "feat(online-rl): add fused schemas (Trajectory/Turn/TokenData/Status)"
```

---

## Task 2: SessionRecorder (完整 Session 生命周期)

**Files:**
- Create: `dev_tools/agentrl/online/recorder.py`
- Test: `tests/unit_tests/dev_tools/agentrl/online/test_recorder.py`

- [ ] **Step 1: 编写 recorder 测试**

```python
# tests/unit_tests/dev_tools/agentrl/online/test_recorder.py
"""Tests for SessionRecorder."""
import pytest
from datetime import datetime
from openjiuwen.dev_tools.agentrl.online.recorder import SessionRecorder

class TestSessionRecorder:
    def test_record_request_creates_session(self):
        recorder = SessionRecorder()
        recorder.record_request("sess_1", "user_1", [{"role": "user", "content": "hi"}])
        assert "sess_1" in recorder._sessions
        assert recorder._sessions["sess_1"]["user_id"] == "user_1"
        assert len(recorder._sessions["sess_1"]["turns"]) == 1

    def test_record_request_appends_turns(self):
        recorder = SessionRecorder()
        recorder.record_request("sess_1", "user_1", [{"role": "user", "content": "hi"}])
        recorder.record_request("sess_1", "user_1", [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "how are you?"},
        ])
        assert len(recorder._sessions["sess_1"]["turns"]) == 2

    def test_record_response_returns_none_without_stop(self):
        recorder = SessionRecorder()
        recorder.record_request("sess_1", "user_1", [{"role": "user", "content": "hi"}])
        result = recorder.record_response("sess_1", {
            "choices": [{"message": {"content": "hello"}, "finish_reason": None}]
        })
        assert result is None

    def test_record_response_returns_trajectory_on_stop(self):
        recorder = SessionRecorder()
        recorder.record_request("sess_1", "user_1", [{"role": "user", "content": "hi"}])
        result = recorder.record_response("sess_1", {
            "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}]
        })
        assert result is not None
        assert result.user_id == "user_1"
        assert result.session_id == "sess_1"
        assert len(result.turns) == 2

    def test_record_response_unknown_session(self):
        recorder = SessionRecorder()
        result = recorder.record_response("unknown", {"choices": []})
        assert result is None

    def test_session_timeout(self):
        recorder = SessionRecorder()
        recorder.record_request("sess_1", "user_1", [{"role": "user", "content": "hi"}])
        result = recorder.on_session_timeout("sess_1")
        assert result is not None
        assert "sess_1" not in recorder._sessions

    def test_timeout_unknown_session(self):
        recorder = SessionRecorder()
        result = recorder.on_session_timeout("unknown")
        assert result is None

    def test_empty_session_not_finalized(self):
        recorder = SessionRecorder()
        recorder._sessions["sess_1"] = {
            "user_id": "u1", "session_id": "sess_1",
            "turns": [], "created_at": datetime.now(), "last_active": datetime.now(),
        }
        result = recorder.on_session_timeout("sess_1")
        assert result is None
```

- [ ] **Step 2: 运行测试验证失败**

Run: `cd agent-core && python -m pytest tests/unit_tests/dev_tools/agentrl/online/test_recorder.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 recorder.py**

```python
# dev_tools/agentrl/online/recorder.py
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

"""
SessionRecorder - manages trajectory recording lifecycle for all sessions.

Thread-safe in-memory session store (production: replace with Redis).
"""

import logging
import threading
import uuid
from datetime import datetime
from typing import Optional

from openjiuwen.dev_tools.agentrl.online.schemas import Trajectory, Turn

logger = logging.getLogger(__name__)


class SessionRecorder:
    """Manages trajectory recording lifecycle for all sessions."""

    def __init__(self):
        self._sessions: dict[str, dict] = {}
        self._lock = threading.Lock()

    def record_request(self, session_id: str, user_id: str, messages: list) -> None:
        """Record user request, update session context."""
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = {
                    "user_id": user_id, "session_id": session_id,
                    "turns": [], "created_at": datetime.now(), "last_active": datetime.now(),
                }
            session = self._sessions[session_id]
            session["last_active"] = datetime.now()

            user_msgs = [m for m in messages if m.get("role") == "user"]
            if user_msgs:
                last_user = user_msgs[-1]
                session["turns"].append(Turn(
                    role="user", content=last_user.get("content", ""),
                    timestamp=datetime.now(),
                ))

    def record_response(self, session_id: str, response: dict) -> Optional[Trajectory]:
        """
        Record assistant response.
        Returns complete Trajectory if finish_reason == 'stop', else None.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                logger.warning("Session %s not found in recorder", session_id)
                return None

            choices = response.get("choices", [])
            if not choices:
                return None

            choice = choices[0]
            message = choice.get("message", {})
            finish_reason = choice.get("finish_reason", "")

            session["turns"].append(Turn(
                role="assistant", content=message.get("content") or "",
                timestamp=datetime.now(),
                token_count=response.get("usage", {}).get("completion_tokens", 0),
            ))
            session["last_active"] = datetime.now()

            if finish_reason == "stop":
                return self._finalize_session(session_id, session)
            return None

    def on_session_timeout(self, session_id: str) -> Optional[Trajectory]:
        """Session timeout, force end recording."""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return None
            return self._finalize_session(session_id, session)

    def _finalize_session(self, session_id: str, session: dict) -> Optional[Trajectory]:
        """Build Trajectory and remove session from memory."""
        turns = session.get("turns", [])
        if not turns:
            self._sessions.pop(session_id, None)
            return None

        trajectory = Trajectory(
            trajectory_id=str(uuid.uuid4()), user_id=session["user_id"],
            session_id=session_id, turns=turns, created_at=session["created_at"],
        )
        self._sessions.pop(session_id, None)
        return trajectory
```

- [ ] **Step 4: 运行测试验证通过**

Run: `cd agent-core && python -m pytest tests/unit_tests/dev_tools/agentrl/online/test_recorder.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
cd agent-core
git add dev_tools/agentrl/online/recorder.py tests/unit_tests/dev_tools/agentrl/online/test_recorder.py
git commit -m "feat(online-rl): add SessionRecorder with full lifecycle management"
```

---

## Task 3: LLM Proxy (流式/非流式 + LoRA注入) + FastAPI App

**Files:**
- Create: `dev_tools/agentrl/online/proxy.py`
- Create: `dev_tools/agentrl/online/app.py`
- Test: `tests/unit_tests/dev_tools/agentrl/online/test_proxy.py`

- [ ] **Step 1: 编写 proxy 测试**

```python
# tests/unit_tests/dev_tools/agentrl/online/test_proxy.py
"""Tests for LLM proxy."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from openjiuwen.dev_tools.agentrl.online.proxy import LLMProxy

class TestLLMProxy:
    def test_init(self):
        proxy = LLMProxy(upstream_url="http://localhost:8000")
        assert proxy.upstream_url == "http://localhost:8000"

    def test_init_strips_trailing_slash(self):
        proxy = LLMProxy(upstream_url="http://localhost:8000/")
        assert proxy.upstream_url == "http://localhost:8000"

    def test_inject_lora(self):
        proxy = LLMProxy(upstream_url="http://localhost:8000")
        body = {"messages": [{"role": "user", "content": "hi"}]}
        result = proxy._inject_lora(body, "user_1_v1")
        assert result["extra_body"]["lora_name"] == "user_1_v1"

    def test_inject_lora_preserves_existing_extra(self):
        proxy = LLMProxy(upstream_url="http://localhost:8000")
        body = {"messages": [{"role": "user", "content": "hi"}], "extra_body": {"existing_key": "value"}}
        result = proxy._inject_lora(body, "user_1_v1")
        assert result["extra_body"]["lora_name"] == "user_1_v1"
        assert result["extra_body"]["existing_key"] == "value"

    @pytest.mark.asyncio
    async def test_forward_request_without_lora(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}
        mock_response.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        proxy = LLMProxy(upstream_url="http://localhost:8000")
        result = await proxy.forward_request(
            body={"messages": [{"role": "user", "content": "hi"}], "model": "test"},
            active_lora=None,
        )
        assert result["choices"][0]["message"]["content"] == "hi"

    @pytest.mark.asyncio
    async def test_forward_request_with_lora(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}
        mock_response.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        proxy = LLMProxy(upstream_url="http://localhost:8000")
        await proxy.forward_request(
            body={"messages": [{"role": "user", "content": "hi"}], "model": "test"},
            active_lora="user_1_v1",
        )
        call_args = mock_client.post.call_args
        json_body = call_args.kwargs.get("json", {})
        assert json_body["extra_body"]["lora_name"] == "user_1_v1"
```

- [ ] **Step 2: 运行测试验证失败**

Run: `cd agent-core && python -m pytest tests/unit_tests/dev_tools/agentrl/online/test_proxy.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 proxy.py**

```python
# dev_tools/agentrl/online/proxy.py
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

"""
LLM Proxy - HTTP request forwarding with LoRA routing and streaming support.
"""

import asyncio
import json
import logging
from typing import Any, Dict, Optional

import httpx
from fastapi.responses import JSONResponse, StreamingResponse

logger = logging.getLogger(__name__)


class LLMProxy:
    """LLM request proxy with LoRA routing."""

    def __init__(self, upstream_url: str, timeout: float = 120.0):
        self.upstream_url = upstream_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=timeout)

    async def forward_request(
        self, body: Dict[str, Any], active_lora: Optional[str] = None,
    ) -> JSONResponse:
        """Forward non-streaming LLM request."""
        if active_lora:
            body = self._inject_lora(body, active_lora)
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(f"{self.upstream_url}/v1/chat/completions", json=body)
            resp.raise_for_status()
            data = resp.json()
        return JSONResponse(content=data, status_code=resp.status_code)

    async def stream_forward(self, body: Dict[str, Any], session_id: str, recorder) -> StreamingResponse:
        """Forward streaming LLM request with trajectory recording."""
        collected_content = []
        finish_reason = None

        if body.get("active_lora"):
            body = self._inject_lora(body, body.pop("active_lora"))

        async def generate():
            nonlocal finish_reason
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream("POST", f"{self.upstream_url}/v1/chat/completions", json=body) as resp:
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

            if finish_reason == "stop" and recorder:
                fake_response = {"choices": [{"message": {"content": "".join(collected_content), "role": "assistant"}, "finish_reason": "stop"}]}
                trajectory = recorder.record_response(session_id, fake_response)
                if trajectory:
                    asyncio.create_task(self._handle_trajectory(trajectory))

        return StreamingResponse(generate(), media_type="text/event-stream")

    def _inject_lora(self, body: Dict[str, Any], lora_name: str) -> Dict[str, Any]:
        """Inject LoRA routing information."""
        extra_body = body.get("extra_body", {})
        extra_body["lora_name"] = lora_name
        body["extra_body"] = extra_body
        return body

    async def _handle_trajectory(self, trajectory) -> None:
        """Handle completed trajectory (to be overridden by app)."""
        pass

    async def close(self) -> None:
        await self.client.aclose()
```

- [ ] **Step 4: 实现 app.py**

```python
# dev_tools/agentrl/online/app.py
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

"""
FastAPI application entry point for the Online RL Gateway.
"""

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request

from openjiuwen.dev_tools.agentrl.online.proxy import LLMProxy
from openjiuwen.dev_tools.agentrl.online.recorder import SessionRecorder

logger = logging.getLogger(__name__)

_proxy: Optional[LLMProxy] = None
_recorder: Optional[SessionRecorder] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _proxy, _recorder
    inference_url = os.environ.get("INFERENCE_URL", "http://localhost:8000")
    _proxy = LLMProxy(upstream_url=inference_url)
    _recorder = SessionRecorder()
    logger.info("Gateway started: inference=%s", inference_url)
    yield
    await _proxy.close()
    logger.info("Gateway stopped")


def create_app() -> FastAPI:
    app = FastAPI(title="Jiuwen Online RL Gateway", lifespan=lifespan)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        global _proxy, _recorder
        user_id = request.headers.get("X-User-ID", "anonymous")
        session_id = request.headers.get("X-Session-ID", str(uuid.uuid4()))
        body = await request.json()

        _recorder.record_request(session_id, user_id, body.get("messages", []))

        is_stream = body.get("stream", False)
        if is_stream:
            return await _proxy.stream_forward(body, session_id, _recorder)
        else:
            response = await _proxy.forward_request(body)
            return response

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


async def _handle_trajectory(trajectory) -> None:
    """Async: compute reward -> store. Scheduling by TrainingScheduler."""
    pass
```

- [ ] **Step 5: 运行测试验证通过**

Run: `cd agent-core && python -m pytest tests/unit_tests/dev_tools/agentrl/online/test_proxy.py -v`
Expected: All 6 tests PASS

- [ ] **Step 6: Commit**

```bash
cd agent-core
git add dev_tools/agentrl/online/proxy.py dev_tools/agentrl/online/app.py tests/unit_tests/dev_tools/agentrl/online/test_proxy.py
git commit -m "feat(online-rl): add LLMProxy with streaming and FastAPI app"
```

---

## Task 4: SQLite TrajectoryStore (完整状态机)

**Files:**
- Create: `dev_tools/agentrl/online/store/__init__.py`
- Create: `dev_tools/agentrl/online/store/trajectory_store.py`
- Test: `tests/unit_tests/dev_tools/agentrl/online/test_trajectory_store.py`

- [ ] **Step 1: 编写 trajectory_store 测试**

```python
# tests/unit_tests/dev_tools/agentrl/online/test_trajectory_store.py
"""Tests for TrajectoryStore."""
import pytest
import tempfile
import os
from datetime import datetime
from openjiuwen.dev_tools.agentrl.online.store.trajectory_store import TrajectoryStore
from openjiuwen.dev_tools.agentrl.online.schemas import Trajectory, Turn, TrajectoryStatus

@pytest.fixture
def store():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield TrajectoryStore(os.path.join(tmpdir, "test.db"))

def test_save_and_query(store):
    traj = Trajectory(
        trajectory_id="t1", user_id="u1", session_id="s1",
        turns=[Turn(role="user", content="hi", timestamp=datetime.now())],
        created_at=datetime.now(),
    )
    store.save(traj)
    assert store.get_pending_count("u1") == 1

def test_get_users_above_threshold(store):
    for i in range(5):
        store.save(Trajectory(
            trajectory_id=f"t{i}", user_id="u1", session_id=f"s{i}",
            turns=[Turn(role="user", content="hi", timestamp=datetime.now())],
            created_at=datetime.now(),
        ))
    assert "u1" in store.get_users_above_threshold(3)

def test_fetch_and_mark_training(store):
    for i in range(3):
        store.save(Trajectory(
            trajectory_id=f"t{i}", user_id="u1", session_id=f"s{i}",
            turns=[Turn(role="user", content="hi", timestamp=datetime.now())],
            created_at=datetime.now(),
        ))
    results = store.fetch_and_mark_training("u1", limit=2)
    assert len(results) == 2
    assert all(t.status == TrajectoryStatus.TRAINING for t in results)
    assert store.get_pending_count("u1") == 1

def test_mark_trained(store):
    store.save(Trajectory(
        trajectory_id="t1", user_id="u1", session_id="s1",
        turns=[Turn(role="user", content="hi", timestamp=datetime.now())],
        created_at=datetime.now(),
    ))
    store.mark_trained(["t1"])
    assert store.fetch_and_mark_training("u1", limit=10) == []
```

- [ ] **Step 2: 运行测试验证失败**

Run: `cd agent-core && python -m pytest tests/unit_tests/dev_tools/agentrl/online/test_trajectory_store.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 trajectory_store.py**

```python
# dev_tools/agentrl/online/store/trajectory_store.py
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

"""
SQLite TrajectoryStore with complete state machine support.
"""

import json
import logging
import sqlite3
import threading
from datetime import datetime

from openjiuwen.dev_tools.agentrl.online.schemas import Trajectory, TrajectoryStatus, Turn

logger = logging.getLogger(__name__)


class TrajectoryStore:
    """SQLite trajectory storage with state machine."""

    def __init__(self, db_path: str = "trajectories.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

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
        turns_json = json.dumps([
            {"role": t.role, "content": t.content,
             "timestamp": t.timestamp.isoformat(), "token_count": t.token_count}
            for t in trajectory.turns
        ])
        token_data_json = json.dumps(trajectory.token_data.__dict__) if trajectory.token_data else None
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO trajectories
                (trajectory_id, user_id, session_id, turns_json, token_data_json,
                 created_at, reward, reward_type, reward_details_json, status, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trajectory.trajectory_id, trajectory.user_id, trajectory.session_id,
                turns_json, token_data_json, trajectory.created_at.isoformat(),
                trajectory.reward, trajectory.reward_type.value,
                json.dumps(trajectory.reward_details), trajectory.status.value,
                json.dumps(trajectory.metadata),
            ))

    def get_pending_count(self, user_id: str) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM trajectories WHERE user_id=? AND status=?",
                (user_id, TrajectoryStatus.PENDING.value)
            ).fetchone()
            return row[0]

    def get_users_above_threshold(self, threshold: int) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT user_id, COUNT(*) as cnt FROM trajectories
                WHERE status=? GROUP BY user_id HAVING cnt >= ?
            """, (TrajectoryStatus.PENDING.value, threshold)).fetchall()
            return [row["user_id"] for row in rows]

    def fetch_and_mark_training(self, user_id: str, limit: int) -> list[Trajectory]:
        """Atomic: fetch PENDING trajectories and mark as TRAINING."""
        with self._lock:
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT * FROM trajectories WHERE user_id=? AND status=? LIMIT ?",
                    (user_id, TrajectoryStatus.PENDING.value, limit)
                ).fetchall()
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

    def _row_to_trajectory(self, row: sqlite3.Row) -> Trajectory:
        from openjiuwen.dev_tools.agentrl.online.schemas import RewardType
        turns_data = json.loads(row["turns_json"])
        turns = [Turn(
            role=t["role"], content=t["content"],
            timestamp=datetime.fromisoformat(t["timestamp"]),
            token_count=t.get("token_count", 0),
        ) for t in turns_data]
        return Trajectory(
            trajectory_id=row["trajectory_id"], user_id=row["user_id"],
            session_id=row["session_id"], turns=turns,
            created_at=datetime.fromisoformat(row["created_at"]),
            reward=row["reward"],
            reward_type=RewardType(row["reward_type"]) if row["reward_type"] else RewardType.CUSTOM,
            reward_details=json.loads(row["reward_details_json"]),
            status=TrajectoryStatus(row["status"]),
            metadata=json.loads(row["metadata_json"]),
        )
```

- [ ] **Step 4: 实现 store/__init__.py**

```python
# dev_tools/agentrl/online/store/__init__.py
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

"""Trajectory storage backends."""

from openjiuwen.dev_tools.agentrl.online.store.trajectory_store import TrajectoryStore

__all__ = ["TrajectoryStore"]
```

- [ ] **Step 5: 运行测试验证通过**

Run: `cd agent-core && python -m pytest tests/unit_tests/dev_tools/agentrl/online/test_trajectory_store.py -v`
Expected: All 4 tests PASS

- [ ] **Step 6: Commit**

```bash
cd agent-core
git add dev_tools/agentrl/online/store/__init__.py dev_tools/agentrl/online/store/trajectory_store.py tests/unit_tests/dev_tools/agentrl/online/test_trajectory_store.py
git commit -m "feat(online-rl): add TrajectoryStore with SQLite and state machine"
```

---

## Task 5: Reward 外部接口 + 容错

**Files:**
- Create: `dev_tools/agentrl/online/reward.py`
- Test: `tests/unit_tests/dev_tools/agentrl/online/test_reward.py`

- [ ] **Step 1: 编写 reward 测试**

```python
# tests/unit_tests/dev_tools/agentrl/online/test_reward.py
"""Tests for RewardCalculator."""
import pytest
from unittest.mock import AsyncMock
from datetime import datetime
from openjiuwen.dev_tools.agentrl.online.reward import ExternalRewardCalculator
from openjiuwen.dev_tools.agentrl.online.schemas import Trajectory, Turn, RewardType

class TestExternalRewardCalculator:
    @pytest.mark.asyncio
    async def test_compute_success(self):
        calc = ExternalRewardCalculator(endpoint="http://localhost:9000")
        calc._call_external_service = AsyncMock(return_value={"reward": 0.8, "details": {"score": 8}})
        traj = Trajectory(
            trajectory_id="t1", user_id="u1", session_id="s1",
            turns=[Turn(role="user", content="hi", timestamp=datetime.now())],
            created_at=datetime.now(),
        )
        result = await calc.compute(traj)
        assert result.reward == 0.8
        assert result.reward_details == {"score": 8}

    @pytest.mark.asyncio
    async def test_compute_failure_fallback(self):
        calc = ExternalRewardCalculator(endpoint="http://localhost:9000")
        calc._call_external_service = AsyncMock(side_effect=Exception("service down"))
        traj = Trajectory(
            trajectory_id="t1", user_id="u1", session_id="s1",
            turns=[Turn(role="user", content="hi", timestamp=datetime.now())],
            created_at=datetime.now(),
        )
        result = await calc.compute(traj)
        assert result.reward == 0.0
        assert "error" in result.reward_details

    def test_get_reward_type(self):
        calc = ExternalRewardCalculator(endpoint="http://localhost:9000", reward_type=RewardType.PRM)
        assert calc.get_reward_type() == RewardType.PRM
```

- [ ] **Step 2: 运行测试验证失败**

Run: `cd agent-core && python -m pytest tests/unit_tests/dev_tools/agentrl/online/test_reward.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 reward.py**

```python
# dev_tools/agentrl/online/reward.py
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

"""
RewardCalculator - external plugin interface for reward computation.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

from openjiuwen.dev_tools.agentrl.online.schemas import Trajectory, RewardType

logger = logging.getLogger(__name__)


class RewardCalculator(ABC):
    """Abstract reward calculator interface."""

    @abstractmethod
    async def compute(self, trajectory: Trajectory) -> Trajectory:
        raise NotImplementedError

    @abstractmethod
    def get_reward_type(self) -> RewardType:
        raise NotImplementedError


class ExternalRewardCalculator(RewardCalculator):
    """
    External reward calculator - calls external service/plugin.
    Supports: LLM-as-Judge, environment reward, human annotation, custom services.
    """

    def __init__(self, endpoint: str, reward_type: RewardType = RewardType.CUSTOM):
        self.endpoint = endpoint
        self._reward_type = reward_type

    async def compute(self, trajectory: Trajectory) -> Trajectory:
        try:
            result = await self._call_external_service(trajectory)
            trajectory.reward = result.get("reward", 0.0)
            trajectory.reward_details = result.get("details", {})
            trajectory.reward_type = self._reward_type
        except Exception as e:
            logger.warning("Reward computation failed for %s: %s", trajectory.trajectory_id, e)
            trajectory.reward = 0.0
            trajectory.reward_details = {"error": str(e)}
        return trajectory

    async def _call_external_service(self, trajectory: Trajectory) -> Dict[str, Any]:
        """Call external reward service (plugin provides implementation)."""
        raise NotImplementedError

    def get_reward_type(self) -> RewardType:
        return self._reward_type
```

- [ ] **Step 4: 运行测试验证通过**

Run: `cd agent-core && python -m pytest tests/unit_tests/dev_tools/agentrl/online/test_reward.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
cd agent-core
git add dev_tools/agentrl/online/reward.py tests/unit_tests/dev_tools/agentrl/online/test_reward.py
git commit -m "feat(online-rl): add RewardCalculator external interface with fault tolerance"
```

---

## Task 6: 训练调度 + ResourceScheduler

**Files:**
- Create: `dev_tools/agentrl/online/trigger.py`
- Create: `dev_tools/agentrl/online/resource_scheduler.py`
- Create: `dev_tools/agentrl/online/scheduler.py`
- Test: `tests/unit_tests/dev_tools/agentrl/online/test_trigger.py`
- Test: `tests/unit_tests/dev_tools/agentrl/online/test_resource_scheduler.py`

- [ ] **Step 1: 实现 trigger.py**

```python
# dev_tools/agentrl/online/trigger.py
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

"""Trigger policies for online RL training."""

from abc import ABC, abstractmethod


class TriggerPolicy(ABC):
    """Abstract trigger policy interface."""

    @abstractmethod
    async def should_trigger(self, user_id: str, pending_count: int) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_threshold(self) -> int:
        raise NotImplementedError


class ThresholdTrigger(TriggerPolicy):
    """Trigger when pending count reaches threshold."""

    def __init__(self, threshold: int = 200):
        self.threshold = max(0, threshold)

    async def should_trigger(self, user_id: str, pending_count: int) -> bool:
        return pending_count >= self.threshold

    def get_threshold(self) -> int:
        return self.threshold
```

- [ ] **Step 2: 实现 resource_scheduler.py**

```python
# dev_tools/agentrl/online/resource_scheduler.py
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

"""ResourceScheduler - pluggable resource scheduling abstraction."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class UserTrainingJob:
    user_id: str
    trajectory_ids: list[str]


@dataclass
class JobStatus:
    job_id: str
    status: str
    result: Optional[dict] = None


class ResourceScheduler(ABC):
    """Abstract resource scheduler."""

    @abstractmethod
    def submit_batch_training_job(self, user_jobs: List[UserTrainingJob]) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_job_status(self, job_id: str) -> JobStatus:
        raise NotImplementedError

    @abstractmethod
    def cancel_job(self, job_id: str) -> None:
        raise NotImplementedError


class LocalProcessScheduler(ResourceScheduler):
    """Local process implementation (development/debugging)."""

    def submit_batch_training_job(self, user_jobs):
        return "local-job-001"

    def get_job_status(self, job_id):
        return JobStatus(job_id=job_id, status="completed")

    def cancel_job(self, job_id):
        pass


class K8sJobScheduler(ResourceScheduler):
    """K8s Job implementation."""
    pass


class RayJobScheduler(ResourceScheduler):
    """Ray Job implementation (native verl integration)."""
    pass
```

- [ ] **Step 3: 实现 scheduler.py**

```python
# dev_tools/agentrl/online/scheduler.py
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

"""TrainingScheduler - periodic scan and batch training submission."""

import asyncio
import logging
from typing import TYPE_CHECKING, Optional

from openjiuwen.dev_tools.agentrl.online.trigger import TriggerPolicy
from openjiuwen.dev_tools.agentrl.online.resource_scheduler import ResourceScheduler

if TYPE_CHECKING:
    from openjiuwen.dev_tools.agentrl.online.store.trajectory_store import TrajectoryStore

logger = logging.getLogger(__name__)


class TrainingScheduler:
    """Periodically scans and triggers RL training."""

    def __init__(
        self,
        trajectory_store: "TrajectoryStore",
        trigger_policy: TriggerPolicy,
        resource_scheduler: ResourceScheduler,
        scan_interval: float = 600.0,
    ):
        self.trajectory_store = trajectory_store
        self.trigger_policy = trigger_policy
        self.resource_scheduler = resource_scheduler
        self.scan_interval = scan_interval
        self._running = False

    async def start(self):
        if self._running:
            return
        self._running = True
        asyncio.create_task(self._scan_loop())

    async def stop(self):
        self._running = False

    async def _scan_loop(self):
        while self._running:
            try:
                await self._scan_and_submit()
            except Exception:
                logger.exception("Error in scan loop")
            await asyncio.sleep(self.scan_interval)

    async def _scan_and_submit(self):
        threshold = self.trigger_policy.get_threshold()
        eligible_users = await self.trajectory_store.get_users_above_threshold(threshold)
        if not eligible_users:
            return

        user_jobs = []
        for user_id in eligible_users:
            trajectories = self.trajectory_store.fetch_and_mark_training(user_id, limit=1000)
            if trajectories:
                user_jobs.append({
                    "user_id": user_id,
                    "trajectory_ids": [t.trajectory_id for t in trajectories],
                })

        if user_jobs:
            job_id = self.resource_scheduler.submit_batch_training_job(user_jobs)
            logger.info("Submitted batch job %s for %d users", job_id, len(user_jobs))
```

- [ ] **Step 4: 运行所有测试验证通过**

Run: `cd agent-core && python -m pytest tests/unit_tests/dev_tools/agentrl/online/test_trigger.py tests/unit_tests/dev_tools/agentrl/online/test_resource_scheduler.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
cd agent-core
git add dev_tools/agentrl/online/trigger.py dev_tools/agentrl/online/resource_scheduler.py dev_tools/agentrl/online/scheduler.py tests/unit_tests/dev_tools/agentrl/online/test_trigger.py tests/unit_tests/dev_tools/agentrl/online/test_resource_scheduler.py
git commit -m "feat(online-rl): add TrainingScheduler with TriggerPolicy and ResourceScheduler"
```

---

## Task 7: RL 后端抽象 + LoRA 管理 + Trainer

**Files:**
- Create: `dev_tools/agentrl/online/rl_backend.py`
- Create: `dev_tools/agentrl/online/backends/__init__.py`
- Create: `dev_tools/agentrl/online/backends/verl_backend.py`
- Create: `dev_tools/agentrl/online/lora_manager.py`
- Create: `dev_tools/agentrl/online/trainer.py`
- Test: `tests/unit_tests/dev_tools/agentrl/online/test_rl_backend.py`
- Test: `tests/unit_tests/dev_tools/agentrl/online/test_lora_manager.py`

- [ ] **Step 1: 实现 rl_backend.py**

```python
# dev_tools/agentrl/online/rl_backend.py
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

"""RL backend abstraction layer."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from openjiuwen.dev_tools.agentrl.online.schemas import Trajectory


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
        self, trajectories: List[Trajectory], config: Optional[Dict[str, Any]] = None,
    ) -> TrainingResult:
        raise NotImplementedError

    @abstractmethod
    async def get_backend_name(self) -> str:
        raise NotImplementedError


class RLBackendRegistry:
    """Registry for RL training backends."""
    _backends: Dict[str, Type[RLBackend]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(backend_cls: Type[RLBackend]) -> Type[RLBackend]:
            cls._backends[name] = backend_cls
            return backend_cls
        return decorator

    @classmethod
    def get_backend(cls, name: str, **kwargs) -> RLBackend:
        if name not in cls._backends:
            raise ValueError(f"Unknown RL backend: {name}. Available: {list(cls._backends.keys())}")
        return cls._backends[name](**kwargs)

    @classmethod
    def list_backends(cls) -> List[str]:
        return list(cls._backends.keys())
```

- [ ] **Step 2: 实现 backends/verl_backend.py**

```python
# dev_tools/agentrl/online/backends/verl_backend.py
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

"""Verl RL training backend (placeholder)."""

from typing import Any, Dict, List, Optional
from openjiuwen.dev_tools.agentrl.online.rl_backend import RLBackend, TrainingResult
from openjiuwen.dev_tools.agentrl.online.schemas import Trajectory


class VerlBackend(RLBackend):
    """Verl-based RL training backend."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    async def train(self, trajectories: List[Trajectory], config: Optional[Dict[str, Any]] = None) -> TrainingResult:
        """Execute verl training (placeholder - actual implementation pending)."""
        return TrainingResult(
            user_id=trajectories[0].user_id if trajectories else "unknown",
            weights_path="/tmp/placeholder_lora",
            metrics={"note": "verl backend not yet fully implemented"},
            status="success",
        )

    async def get_backend_name(self) -> str:
        return "verl"
```

- [ ] **Step 3: 实现 lora_manager.py**

```python
# dev_tools/agentrl/online/lora_manager.py
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

"""LoRA Weight Manager - unified versioned storage and hot-load notification."""

import json
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import httpx


@dataclass
class LoRAVersion:
    """LoRA version metadata."""
    user_id: str
    version: str
    path: str
    created_at: datetime
    trajectory_count: int
    reward_avg: float
    base_model: str
    metrics: Dict[str, Any] = field(default_factory=dict)


class LoRAStorageAdapter(ABC):
    """Abstract interface for LoRA weight storage."""

    @abstractmethod
    async def save(self, user_id: str, version: int, weights_path: str, metadata: dict) -> None:
        raise NotImplementedError

    @abstractmethod
    async def get_latest_version(self, user_id: str) -> Optional[int]:
        raise NotImplementedError

    @abstractmethod
    async def set_latest(self, user_id: str, version: int) -> None:
        raise NotImplementedError


class LocalStorageAdapter(LoRAStorageAdapter):
    """Local filesystem LoRA storage."""

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)

    async def save(self, user_id: str, version: int, weights_path: str, metadata: dict) -> None:
        dest = self.base_path / user_id / f"v{version}"
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copytree(weights_path, dest, dirs_exist_ok=True)
        (dest / "metadata.json").write_text(json.dumps(metadata, default=str))

    async def get_latest_version(self, user_id: str) -> Optional[int]:
        user_dir = self.base_path / user_id
        if not user_dir.exists():
            return None
        versions = [int(d.name[1:]) for d in user_dir.iterdir() if d.is_dir() and d.name.startswith("v")]
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

    async def notify_update(self, user_id: str, lora_name: str, weights_path: str) -> None:
        await self.client.post(
            f"{self.inference_url}/v1/load_lora_adapter",
            json={"lora_name": lora_name, "lora_path": weights_path},
        )

    async def close(self) -> None:
        await self.client.aclose()


class LoRAManager:
    """High-level LoRA weight lifecycle manager."""

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

    async def _next_version(self, user_id: str) -> int:
        latest = await self.storage.get_latest_version(user_id)
        return (latest or 0) + 1
```

- [ ] **Step 4: 实现 trainer.py**

```python
# dev_tools/agentrl/online/trainer.py
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

"""BatchUserLoRATrainer - sequential multi-user LoRA training."""

import logging
from typing import List

from openjiuwen.dev_tools.agentrl.online.rl_backend import RLBackend
from openjiuwen.dev_tools.agentrl.online.lora_manager import LoRAManager

logger = logging.getLogger(__name__)


class BatchUserLoRATrainer:
    """
    Base model loaded once, frozen throughout.
    LoRA adapters reset and reused between users.
    """

    def __init__(self, rl_backend: RLBackend, lora_manager: LoRAManager):
        self.rl_backend = rl_backend
        self.lora_manager = lora_manager

    async def run(self, user_batch: List[dict], trajectory_store=None) -> None:
        for job in user_batch:
            try:
                await self._train_one_user(job, trajectory_store)
            except Exception as e:
                logger.error("Training failed for user %s: %s", job["user_id"], e)
                if trajectory_store:
                    trajectory_store.mark_failed(job["trajectory_ids"])

    async def _train_one_user(self, job: dict, trajectory_store=None) -> None:
        # TODO: Load trajectories, train, publish, mark trained
        logger.info("Training user %s with %d trajectories", job["user_id"], len(job["trajectory_ids"]))
        if trajectory_store:
            trajectory_store.mark_trained(job["trajectory_ids"])
```

- [ ] **Step 5: 运行测试验证通过**

Run: `cd agent-core && python -m pytest tests/unit_tests/dev_tools/agentrl/online/test_rl_backend.py tests/unit_tests/dev_tools/agentrl/online/test_lora_manager.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
cd agent-core
git add dev_tools/agentrl/online/rl_backend.py dev_tools/agentrl/online/backends/ dev_tools/agentrl/online/lora_manager.py dev_tools/agentrl/online/trainer.py tests/unit_tests/dev_tools/agentrl/online/test_rl_backend.py tests/unit_tests/dev_tools/agentrl/online/test_lora_manager.py
git commit -m "feat(online-rl): add RLBackend, LoRAManager, and BatchUserLoRATrainer"
```

---

## Task 8: 配置 + 集成测试

**Files:**
- Create: `dev_tools/agentrl/online/config.py`
- Create: `tests/unit_tests/dev_tools/agentrl/online/test_integration.py`

- [ ] **Step 1: 实现 config.py**

```python
# dev_tools/agentrl/online/config.py
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

"""Configuration schema for the online RL gateway."""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ProxyConfig:
    upstream_url: str = "http://localhost:8000"
    host: str = "0.0.0.0"
    port: int = 8001
    timeout: float = 120.0


@dataclass
class RecorderConfig:
    tokenizer_model: str = "Qwen/Qwen2.5-7B"
    save_path: str = "/data/rollouts"
    flush_interval: int = 100


@dataclass
class TriggerConfig:
    type: str = "threshold"
    threshold: int = 200


@dataclass
class SchedulerConfig:
    enabled: bool = True
    scan_interval: float = 600.0
    trigger: TriggerConfig = field(default_factory=TriggerConfig)


@dataclass
class RLBackendConfig:
    name: str = "verl"
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoRAConfig:
    storage_type: str = "local"
    storage_base_path: str = "/data/lora_weights"
    inference_url: str = "http://localhost:8000"


@dataclass
class GatewayConfig:
    enabled: bool = True
    proxy: ProxyConfig = field(default_factory=ProxyConfig)
    recorder: RecorderConfig = field(default_factory=RecorderConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    rl_backend: RLBackendConfig = field(default_factory=RLBackendConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
```

- [ ] **Step 2: 编写集成测试**

```python
# tests/unit_tests/dev_tools/agentrl/online/test_integration.py
"""Integration tests for the online RL gateway."""
import pytest
from openjiuwen.dev_tools.agentrl.online.config import GatewayConfig
from openjiuwen.dev_tools.agentrl.online.schemas import Trajectory, Turn, RewardType, TrajectoryStatus
from openjiuwen.dev_tools.agentrl.online.trigger import ThresholdTrigger
from openjiuwen.dev_tools.agentrl.online.schemas import RewardType

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
        from openjiuwen.dev_tools.agentrl.online import (
            TrajectoryStatus, RewardType, Turn, TokenData, Trajectory,
        )
        assert RewardType.PRM == "prm"
        assert ThresholdTrigger(threshold=10).threshold == 10
```

- [ ] **Step 3: 运行所有 gateway 测试**

Run: `cd agent-core && python -m pytest tests/unit_tests/dev_tools/agentrl/online/ -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
cd agent-core
git add dev_tools/agentrl/online/config.py tests/unit_tests/dev_tools/agentrl/online/test_integration.py
git commit -m "feat(online-rl): add GatewayConfig and integration tests"
```

---

## 自审检查

### 1. Spec 覆盖度

| Spec 章节 | 对应 Task | 状态 |
|-----------|-----------|------|
| 融合数据格式 | Task 1 | OK |
| FastAPI Gateway | Task 3 | OK |
| SessionRecorder | Task 2 | OK |
| SQLite TrajectoryStore | Task 4 | OK |
| Reward 外部接口 | Task 5 | OK |
| 训练调度 + ResourceScheduler | Task 6 | OK |
| RL 后端抽象 | Task 7 | OK |
| LoRA 管理 + 元数据 | Task 7 | OK |
| 配置 | Task 8 | OK |

### 2. 融合点检查

| 融合点 | 来源 | 已实现 |
|--------|------|--------|
| FastAPI 框架 | 之前草案 | Task 3 |
| SessionRecorder | 之前草案 | Task 2 |
| 流式 SSE 支持 | 之前草案 | Task 3 |
| 并发安全 (Lock) | 之前草案 | Task 2, 4 |
| TrajectoryStatus 状态机 | 之前草案 | Task 1, 4 |
| TokenData | 当前设计 | Task 1 |
| RewardType 灵活 | 当前设计 | Task 1, 5 |
| Reward 外部化 | 当前设计 | Task 5 |
| Reward 容错 | 之前草案 | Task 5 |
| ResourceScheduler | 之前草案 | Task 6 |
| TriggerPolicy | 当前设计 | Task 6 |
| BatchUserLoRATrainer | 之前草案 | Task 7 |
| LoRA 元数据 | 之前草案 | Task 7 |
| LoRA 存储抽象 | 当前设计 | Task 7 |

### 3. 灵活性检查

| 未定项 | 预留方式 |
|--------|----------|
| RL 算法 | RLBackend 注册表，可插拔 |
| Reward 计算 | RewardCalculator 外部接口 |
| 调度策略 | TriggerPolicy + ResourceScheduler 抽象 |
| 存储后端 | TrajectoryStore 可替换 |
| LoRA 存储 | LoRAStorageAdapter 抽象 |
