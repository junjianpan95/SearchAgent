# SearchAgent

<div align="center">

**基于在线检索与多策略后训练的 Agentic RAG 多跳问答系统**

*对标 AgentRL（基于证据链课程强化学习的 Agentic RAG），实现「结构化动作空间 + 混合检索后端 + {LoRA, DoRA} × {GRPO, GSPO} 四方案横向对比」*

</div>

---

## 📖 项目简介

SearchAgent 是一个面向**多跳问答（Multi-hop QA）**的 Agentic RAG 系统，核心思路是让语言模型在**结构化动作空间**中通过「检索 → 阅读 → 归因 → 作答」的多步推理，逐步定位并串联分散在多个文档中的证据，最终给出可溯源、有引用支撑的答案。

与「单轮 RAG」不同，本项目的 Agent 会**主动规划检索策略**（搜索、阅读、补充证据），并通过**课程化强化学习**（SFT → GRPO/GSPO）训练模型学会「何时搜、搜什么、何时停」。

### 相对 AgentRL 的两处关键差异化

| 维度 | AgentRL（基线） | SearchAgent（本项目） |
|------|-----------------|----------------------|
| **搜索后端** | 单一在线检索 | 本地混合索引（BM25 + BGE-M3 + RRF + Reranker）为主，`Tavily` 在线搜索可插拔，支持「在线 vs 本地」消融对比 |
| **训练方案** | 单一 RL 方案 | 对 `{LoRA, DoRA} × {GRPO, GSPO}` 四套后训练方案做同基座、同评测集的横向对比并择优 |

---

## ✨ 核心特性

- **结构化动作空间**：`Search / Read / AddEvidence / Answer` 四类动作，输入输出均可解析、可评估
- **Agent 记忆**：固定容量 Evidence Memory、历史查询去重、桥接实体 direct-read、结构化 recovery feedback
- **预算约束**：最多 4 次 Search、6 次 Read、8K token 预算，防止无限检索
- **两阶段课程奖励**：Stage 1 优化新增 supporting-fact / 引用有效性 / grounding；Stage 2 强化 Answer EM/F1 并保留证据约束
- **混合检索后端**：BM25 + BGE-M3 稠密向量 + RRF 融合，可选 BGE-Reranker 重排
- **双模式运行**：`local_smoke`（单卡 16G 可跑通全链路）+ `server_full`（多卡全量出真实数字），同一套代码 `--config` 切换

---

## 🧠 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                        SearchAgent                           │
│                                                              │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐  │
│  │  Search  │ → │   Read   │ → │ AddEvid  │ → │  Answer  │  │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘  │
│       ↓              ↓              ↓                         │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Evidence Memory（证据记忆）              │    │
│  └─────────────────────────────────────────────────────┘    │
│       ↓                                                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │         混合检索后端（Hybrid Retrieval）              │    │
│  │   BM25 稀疏 ──┐                                       │    │
│  │   BGE-M3 稠密 ─┼── RRF 融合 ── Reranker 重排         │    │
│  │   Tavily 在线 ─┘                                       │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 快速开始

### 环境说明（重要）

本项目**不使用 conda**，而是项目自带的 Python 虚拟环境：

```
E:\Life\1\SearchAgent\.venv\Scripts\python.exe
```

所有脚本都会自动优先使用这个解释器。若要用命令行手动激活：

```powershell
# PowerShell
E:\Life\1\SearchAgent\.venv\Scripts\Activate.ps1
# CMD
E:\Life\1\SearchAgent\.venv\Scripts\activate.bat
```

### 在线问答（推荐：一键启动前后端）

```powershell
# 方式一：双击 run.bat —— 自动开两个窗口（后端 8001 + 前端 8000）
# 方式二：手动开两个窗口
backend.bat           # 窗口1：在线问答后端（规则模式，秒启动，无需 GPU）
start.bat             # 窗口2：前端页面
```

启动后浏览器打开 <http://127.0.0.1:8000>，在「在线问答」页输入问题，Agent 会实时执行 Search → Read → AddEvidence → Answer 多跳检索。

- `backend.bat`（规则模式）：用 BM25 检索 + 规则化动作序列，**秒启动、零模型依赖**，适合快速体验
- `backend.bat --model`（模型模式）：加载 Qwen3-1.7B 真实推理（需先下载模型，首次加载较慢）
- 后端健康检查：<http://127.0.0.1:8001/api/health>

### 前端 Demo 启停（Windows）

| 方式 | 启动 | 停止 |
|------|------|------|
| **一键启动** | 双击 `run.bat`（前端+后端） | 关闭两个窗口，或 `stop.bat` |
| 仅前端 | 双击 `start.bat`（前台窗口） | 在该窗口按 `Ctrl+C` |
| 仅后端 | 双击 `backend.bat` | 在该窗口按 `Ctrl+C` |
| PowerShell | `.\start.ps1` | `.\stop.ps1` |

默认端口：前端 `8000`、后端 `8001`。自定义端口：`start.bat 9000`。

### 命令行全链路（Git Bash / WSL）

```bash
# 1. 安装依赖（已有 venv 则跳过）
bash scripts/setup.sh

# 2. 下载模型（Qwen3-1.7B + BGE-M3，走 ModelScope 国内源）
python scripts/download_models.py --all

# 3. 下载数据集（AgenticRAGTracer + HotpotQA + MuSiQue）
python scripts/download_dataset.py

# 4. 构建全量语料 + 混合索引（BM25 + BGE-M3 + RRF）
python scripts/build_corpus.py
python scripts/build_index.py

# 5. 构造轨迹（AgenticRagTrace 1305 条 → 60/25/15 三类轨迹）
python -m src.data_gen.build_data --config configs/local_smoke.yaml --n 1305

# 6. SFT 训练（LLaMA-Factory LoRA / DoRA）
.\.venv\Scripts\python.exe -m llamafactory.cli train configs/llamafactory_lora.yaml
.\.venv\Scripts\python.exe -m llamafactory.cli train configs/llamafactory_dora.yaml

# 7. 四方案横向评测
python scripts/run_compare.py --n 15 --methods base_react lora dora

# 8. verl GRPO/GSPO（WSL2 内执行）
bash scripts/setup_wsl.sh            # 搭 WSL 环境
bash scripts/run_verl_grpo.sh grpo   # 或 gspo
```

### Agent 端到端 Demo（无需下载大模型）

在终端直接观察 SearchAgent 的完整多跳推理过程：

```powershell
# Windows（项目根目录）
.\.venv\Scripts\python.exe scripts\demo_agent.py --config configs\local_smoke.yaml --no-dense
```

- `--no-dense`：只用 BM25 检索，**不需要下载 BGE-M3 大模型**，秒级跑通
- 去掉 `--no-dense`：启用 BM25 + BGE-M3 + RRF 混合检索（需先下载模型）
- 输出：每道多跳题的 `Search → AddEvidence → Answer` 全过程 + 最终答案 + 引用编号
- 轨迹保存到 `data/demo_trajectories.json`（前端页面从这里取数据渲染）

---

## 📊 实验结果

> 以下为本地 RTX 5060 Ti 16G + Qwen3-1.7B 的真实跑通数字，评测集为页面隔离的 sealed test（100 条）。

### 五方案横向对比（sealed test, n=100）

| 方案 | Answer F1 | Joint F1 | 证据链 All-Hit | 冗余率 | 平均工具调用 |
|------|-----------|----------|---------------|--------|-------------|
| Base ReAct（规则基线） | 0.0201 | 0.0375 | 1.0000 | 0.0000 | 2.00 |
| **LoRA** | **0.0387** | **0.0477** | 0.4967 | 0.0000 | 1.92 |
| DoRA | 0.0292 | 0.0365 | 0.4992 | 0.0000 | 1.96 |
| LoRA + GRPO | 0.0307 | 0.0000 | 0.0000 | 0.0000 | 0.00 |
| LoRA + GSPO | 0.0306 | 0.0000 | 0.0000 | 0.0000 | 0.00 |

### 关键结论

1. **SFT 带来显著提升**：LoRA 将 Answer F1 从 Base ReAct 的 0.0201 提升至 0.0387（×1.92），Joint F1 从 0.0375 提升至 0.0477。
2. **DoRA 略逊于 LoRA**：在同量级参数下，DoRA 的 Answer F1（0.0292）略低于 LoRA（0.0387），但证据链 All-Hit 相近。
3. **RL 后训练（GRPO/GSPO）出现退化**：RL 后模型在贪心解码下退化为「凭记忆直接作答」，检索能力归零（工具调用 0 次），Answer F1 回落至 0.0307/0.0306。这是一个**有价值的负结果**，根因已定位为奖励对 `<think>` 前缀的隐形容忍 + 组内优势归零 + KL 约束过弱（详见 `docs/项目介绍.md` 第 7.6.2 节与「坑 16」）。

> 关于数据泄漏：过程中通过集合交集校验发现并修复了「训练集覆盖评测集」的数据泄漏，实测泄漏会虚高 27%–59% 并翻转 LoRA/DoRA 的优劣排序。上表数据均已隔离泄漏。

---

## 📁 目录结构

```
configs/         双模式配置（local_smoke / server_full）+ LLaMA-Factory 训练配置
src/agent/       SearchAgent 核心（动作空间 / 记忆 / 预算 / 停止策略）
src/retrieval/   混合检索后端（BM25 + BGE-M3 + RRF + Reranker，Tavily 可插拔）
src/data_gen/    SearchTrace 轨迹构造（AgenticRagTrace 多跳数据集生成）
src/train/       SFT(LoRA/DoRA) + GRPO/GSPO（verl/TRL）训练
src/reward/      两阶段课程奖励
src/utils/       指标计算等工具
evals/           四方案横向评测
web/             前端可视化 Demo
scripts/         一键启停、模型/数据下载、构建、训练、评测脚本
docs/            项目介绍（含踩坑复盘）与 WSL 重跑指南
```

> 说明：`data/` 目录下的模型权重、数据集、索引、轨迹缓存等大文件**未纳入版本管理**（见 `.gitignore`），需按「快速开始」自行下载。

---

## 🔧 硬件要求

| 模式 | 基座 | 训练框架 | 硬件 |
|------|------|---------|------|
| local_smoke | Qwen3-1.7B | LoRA SFT + TRL GRPO | 单卡 16GB（RTX 5060 Ti） |
| server_full | Qwen3-8B | LoRA/DoRA SFT + verl/vLLM/Ray/FSDP | 多卡 A100/H100 |

---

## 📚 文档

- **项目介绍**（含完整踩坑复盘与心路历程）：`docs/项目介绍.md`
- **WSL + verl 重跑指南**：`docs/WSL_verl_重跑指南.md`

---

## 📄 License

MIT License
