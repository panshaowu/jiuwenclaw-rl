# Mermaid 图表渲染指南

本目录包含在线RL框架设计文档的所有 Mermaid 图表源码。

## 图表清单

| 文件 | 名称 | 类型 | 来源文档 |
|------|------|------|---------|
| `01-component-relationships.mmd` | 组件关系图 | graph TB | Section 3.2 |
| `02-workflow-overview.mmd` | 工作流全景图 | flowchart TB | Section 3.3.1 |
| `03-dependency-graph.mmd` | 依赖关系图 | graph TB | Section 3.5.2 |
| `04-sequence-diagram.mmd` | 请求处理序列图 | sequenceDiagram | Section 3.6.1 |
| `05-data-flow.mmd` | 数据流视图 | flowchart TD | Section 3.6.2 |

## 渲染方式

### 方式1: 在线渲染 (推荐)

访问 [Mermaid Live Editor](https://mermaid.live/)，将 `.mmd` 文件内容粘贴到编辑器中即可实时预览和导出 PNG/SVG。

### 方式2: VS Code 插件

安装 [Mermaid Preview](https://marketplace.visualstudio.com/items?itemName=bierner.markdown-mermaid) 或 [Markdown Preview Mermaid Support](https://marketplace.visualstudio.com/items?itemName=bierner.markdown-mermaid) 插件，直接在 VS Code 中预览。

### 方式3: 命令行渲染 (需要 Node.js)

```bash
# 安装 mmdc (Mermaid CLI)
npm install -g @mermaid-js/mermaid-cli

# 渲染单个图表为 SVG
mmdc -i 01-component-relationships.mmd -o 01-component-relationships.svg

# 渲染所有图表
for f in *.mmd; do
  mmdc -i "$f" -o "${f%.mmd}.svg" -t default -b white
done
```

### 方式4: PlantUML (需要 Java)

如果已安装 Java 和 PlantUML jar:

```bash
# PlantUML 不直接支持 Mermaid 语法
# 需要先将 Mermaid 转换为 PlantUML 或使用其他方式
# 推荐使用方式1或方式3
```

## 配色方案

| 配色 | 含义 | 说明 |
|------|------|------|
| 🟢 绿色 `#d4edda` | 在线RL特有 | 运行时服务，Agent运行期间持续工作 |
| 🔵 蓝色 `#d1ecf1` | RL公共基础 | 在线/离线共用，提供核心RL能力 |
| 🟠 橙色 `#fff3cd` | 离线RL特有 | 开发/训练工具，开发者手动触发 |
| ⬜ 灰色 `#e2e3e5` | 外部服务 | verl、vLLM等第三方依赖 |

## 修改图表

1. 编辑对应的 `.mmd` 文件
2. 使用上述任一方式预览效果
3. 同步更新设计文档 `2026-03-31-online-rl-framework-design-v3.md` 中的对应图表
