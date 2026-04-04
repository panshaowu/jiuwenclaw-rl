# Jiuwen Online RL Framework Documentation

This repository contains design documents and implementation plans for the Jiuwen Agent-Core online reinforcement learning framework.

## Overview

The online RL framework enables AI agents to learn continuously while interacting with users, without requiring service downtime.

**V4 Architecture** (Latest): Pure SDK mode integrated into `agent_evolving` module as part of self-evolution capabilities:
- **LLMAgentProxy**: In-process proxy for trajectory collection via RAIL hooks
- **RolloutCollector**: Core component managing trajectory collection workflow
- **TrajectoryStore**: SQLite-based storage with state machine
- **TrainingScheduler**: Pluggable trigger policies for automated training
- **RLBackend**: Abstraction layer supporting verl and other frameworks
- **LoRAManager**: Unified weight versioning and hot-loading
- **EvolutionOrchestrator**: Unified management of rule-based and RL-based evolution strategies

**V3 Architecture** (Legacy): Service-based with FastAPI gateway.

## Document Structure

### 📋 [Specs](./specs/)
Design documents describing the framework architecture:
- [v1.0](./specs/2026-03-31-online-rl-framework-design.md) - Initial design (superseded)
- [v2.0](./specs/2026-03-31-online-rl-framework-design-v2.md) - Fused version (superseded)
- [v3.0](./specs/2026-03-31-online-rl-framework-design-v3.md) - Service-based design (superseded)
- [v4.0](./specs/2026-04-03-online-rl-framework-design-v4.md) - **Current** pure SDK mode design

### 📝 [Plans](./plans/)
Step-by-step implementation guides:
- [v1.0](./plans/2026-03-31-online-rl-framework.md) - Initial plan
- [v2.0](./plans/2026-03-31-online-rl-framework-v2.md) - Fused version plan
- [v3.0](./plans/2026-03-31-online-rl-framework-v3.md) - Service-based implementation plan
- [v4.0](./plans/2026-04-03-online-rl-framework-v4.md) - **Current** pure SDK implementation plan

### 📊 [Diagrams](./diagrams/)
Mermaid diagrams visualizing the architecture:
- Component relationships
- Workflow overview
- Dependency graph
- Sequence diagrams
- Data flow diagrams

### 📝 [Reviews](./reviews/)
Design analysis and review documents:
- [V3 Design Issues](./reviews/2026-04-01-v3-design-issues.md) - Architectural problems in V3
- [Evolution Coordination](./reviews/2026-04-03-evolution-coordination-analysis.md) - Rule engine vs RL trigger coordination
- [Multi-App Sharing](./reviews/2026-04-03-multi-app-sharing-analysis.md) - Multi-application scenarios analysis
- [V4 Final Review](./reviews/2026-04-03-v4-design-final-review.md) - Final design review checklist
- [V4 Red Team Analysis](./reviews/2026-04-03-v4-design-red-team-analysis.md) - Adversarial design evaluation

## Quick Start

**Latest Design**: See [specs/2026-04-03-online-rl-framework-design-v4.md](./specs/2026-04-03-online-rl-framework-design-v4.md)

**Implementation Guide**: See [plans/2026-04-03-online-rl-framework-v4.md](./plans/2026-04-03-online-rl-framework-v4.md)

**Design Reviews**: See [reviews/](./reviews/)

**Architecture Diagrams**: See [diagrams/](./diagrams/)

## Key Features

**V4 Architecture Benefits**:
✅ **Pure SDK Mode**: No independent services, lightweight integration
✅ **Self-Evolution**: Integrated into `agent_evolving` module
✅ **Zero Downtime**: Agents continue serving during training
✅ **Optional Dependencies**: Install with `openjiuwen[online-rl]`
✅ **Flexible Scheduling**: Pluggable trigger policies
✅ **Multi-Backend**: Supports verl, extensible
✅ **Hot Loading**: Dynamic LoRA weight updates
✅ **Evolution Coordination**: Unified management of rule-based and RL-based evolution

**V3 Architecture** (Legacy):
✅ **Production Ready**: FastAPI, SQLite, concurrent-safe
✅ **Service-Based**: Independent RL gateway service

## Version History

| Version | Date | Status | Description |
|---------|------|--------|-------------|
| v1.0 | 2026-03-31 | Superseded | Initial architecture design |
| v2.0 | 2026-03-31 | Superseded | Fused with production features |
| v3.0 | 2026-03-31 | Superseded | Service-based with FastAPI gateway |
| v4.0 | 2026-04-03 | **Current** | Pure SDK mode, integrated into agent_evolving |

## Major Changes in V4

**Architecture Migration**: From service-based (V3) to pure SDK mode (V4)

**Key Improvements**:
1. **No Independent Services**: Removed FastAPI gateway, using in-process LLMAgentProxy
2. **RAIL Hook Integration**: Trajectory collection via RAIL hooks, non-invasive
3. **Self-Evolution Integration**: RL as part of `agent_evolving` module
4. **Optional Dependencies**: Lightweight via `openjiuwen[online-rl]`
5. **EvolutionOrchestrator**: Unified management of rule-based and RL-based evolution
6. **Application-Level Config**: Config managed by upper-layer applications

**Problem Resolution**:
- P1: SDK vs service layer confusion → Pure SDK mode
- P2: Dependency bloat → Optional dependencies
- P3: Responsibility overlap with jiuwenclaw → Clear separation
- P4: RolloutPersistenceAdapter positioning → Placed in `rl/store/`
- P5: Missing config management → Application-level OnlineRLConfig
- P6: Insufficient reuse from external projects → Clarified as reference only

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
