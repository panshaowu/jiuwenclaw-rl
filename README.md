# Jiuwen Online RL Framework Documentation

This repository contains design documents and implementation plans for the Jiuwen Agent-Core online reinforcement learning framework.

## Overview

The online RL framework enables AI agents to learn continuously while interacting with users, without requiring service downtime. The framework integrates:

- **LLM Gateway**: FastAPI service with trajectory collection and LoRA routing
- **SessionRecorder**: Complete session lifecycle management
- **TrajectoryStore**: SQLite-based storage with state machine
- **TrainingScheduler**: Pluggable trigger policies for automated training
- **RLBackend**: Abstraction layer supporting verl and other frameworks
- **LoRAManager**: Unified weight versioning and hot-loading

## Document Structure

### 📋 [Specs](./specs/)
Design documents describing the framework architecture:
- [v1.0](./specs/2026-03-31-online-rl-framework-design.md) - Initial design (superseded)
- [v2.0](./specs/2026-03-31-online-rl-framework-design-v2.md) - Fused version (superseded)
- [v3.0](./specs/2026-03-31-online-rl-framework-design-v3.md) - **Current** production-ready design

### 📝 [Plans](./plans/)
Step-by-step implementation guides:
- [v1.0](./plans/2026-03-31-online-rl-framework.md) - Initial plan
- [v2.0](./plans/2026-03-31-online-rl-framework-v2.md) - Fused version plan
- [v3.0](./plans/2026-03-31-online-rl-framework-v3.md) - **Current** implementation plan

### 📊 [Diagrams](./diagrams/)
Mermaid diagrams visualizing the architecture:
- Component relationships
- Workflow overview
- Dependency graph
- Sequence diagrams
- Data flow diagrams

## Quick Start

**Latest Design**: See [specs/2026-03-31-online-rl-framework-design-v3.md](./specs/2026-03-31-online-rl-framework-design-v3.md)

**Implementation Guide**: See [plans/2026-03-31-online-rl-framework-v3.md](./plans/2026-03-31-online-rl-framework-v3.md)

**Architecture Diagrams**: See [diagrams/](./diagrams/)

## Key Features

✅ **Zero Downtime**: Agents continue serving during training
✅ **Production Ready**: FastAPI, SQLite, concurrent-safe
✅ **Flexible Scheduling**: Pluggable trigger policies
✅ **Multi-Backend**: Supports verl, slime, extensible
✅ **Hot Loading**: Dynamic LoRA weight updates

## Version History

| Version | Date | Status | Description |
|---------|------|--------|-------------|
| v1.0 | 2026-03-31 | Superseded | Initial architecture design |
| v2.0 | 2026-03-31 | Superseded | Fused with production features |
| v3.0 | 2026-03-31 | **Current** | Production-ready with architectural refactoring |

## Related Projects

- **agent-core** (openjiuwen): Python SDK for LLM applications
- **jiuwenclaw**: AI assistant application built on agent-core
- **verl**: RL training engine backend
- **openclaw-rl**: RL training framework (arXiv:2603.10165)
- **MetaClaw**: Online meta-learning framework (arXiv:2603.17187)

## License

Apache-2.0

## Authors

Sisyphus (AI Agent) with guidance from the Jiuwen team
