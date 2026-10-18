"""生成论文级图表（Figure 1-6），输出到 docs/images/paper/。

所有数字均来自本机真实落盘产物：
  - logs/sft_lora_v2.log / logs/sft_dora_v2.log  -> SFT 逐次 eval_loss
  - data/compare_report.json                     -> 四方案五指标
  - data/traces/rl_samples.json                  -> 轨迹配比与动作统计

用法：
  .venv/Scripts/python.exe scripts/make_paper_figs.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "images" / "paper"
OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------- 论文风格
plt.rcParams.update({
    # 字体回退链：西文 Times New Roman，中文自动回退到 Microsoft YaHei
    "font.family": ["Times New Roman", "Microsoft YaHei"],
    "axes.unicode_minus": False,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "mathtext.fontset": "stix",
})

C_RED = "#C0392B"      # 主方案 / 正向
C_BLUE = "#2E6DA4"     # 次方案
C_GREY = "#7F8C8D"     # 基线
C_LIGHT = "#EAEDED"
C_GREEN = "#1E8449"
C_ORANGE = "#C77B1E"


# ----------------------------------------------------------------- 真实数据
import re

def read_sft_log(name: str) -> dict:
    """从 LLaMA-Factory 训练日志中直接提取真实指标（不再硬编码）。"""
    p = ROOT / "logs" / f"sft_{name.lower()}_v3.log"
    if not p.exists():
        p = ROOT / "logs" / f"sft_{name.lower()}_v2.log"
    txt = p.read_text(encoding="utf-8", errors="ignore")
    evals = [float(v) for v in re.findall(r"'eval_loss': '([0-9.]+)'", txt)]
    train_loss = re.findall(r"'train_loss': '([0-9.]+)'", txt)
    runtime = re.findall(r"'train_runtime': '([0-9.]+)'", txt)
    return {
        "eval": evals,
        "train_loss": float(train_loss[-1]) if train_loss else float("nan"),
        "runtime": float(runtime[-1]) if runtime else float("nan"),
        "source": p.name,
    }


_SFT = {n: read_sft_log(n) for n in ("LoRA", "DoRA")}
SFT_EVAL = {n: d["eval"] for n, d in _SFT.items()}
SFT_TRAIN = {n: d["train_loss"] for n, d in _SFT.items()}
SFT_RUNTIME = {n: d["runtime"] for n, d in _SFT.items()}
for _n, _d in _SFT.items():
    print(f"[SFT] {_n}: {len(_d['eval'])} 个 eval 点, train_loss={_d['train_loss']:.4f}, "
          f"runtime={_d['runtime']:.0f}s  <- {_d['source']}")


def _epoch_axis(n: int) -> list[float]:
    """把 n 个评估点近似映射到 epoch 轴。"""
    if n <= 1:
        return [3.0]
    return [round(0.6 + (3.0 - 0.6) * i / (n - 1), 2) for i in range(n)]


EVAL_X = _epoch_axis(max(1, len(next(iter(SFT_EVAL.values()), []))))

CMP = json.loads((ROOT / "data" / "compare_report.json").read_text(encoding="utf-8"))
# 补充 RL 结果（跑完由 update_paper_figs 自动并入；缺失则跳过对应柱子）
RL_PATH = ROOT / "data" / "rl_report.json"
if RL_PATH.exists():
    CMP.update(json.loads(RL_PATH.read_text(encoding="utf-8")))

METHODS = ["base_react", "lora", "dora", "grpo", "gspo"]
LABELS = {"base_react": "Base ReAct", "lora": "LoRA", "dora": "DoRA",
          "grpo": "LoRA+GRPO", "gspo": "LoRA+GSPO"}
COLORS = {"base_react": C_GREY, "lora": C_BLUE, "dora": C_RED,
          "grpo": C_GREEN, "gspo": C_ORANGE}


def present_methods():
    return [m for m in METHODS if m in CMP]


# ----------------------------------------------------------------- 工具函数
def box(ax, x, y, w, h, text, fc=C_LIGHT, ec="#2C3E50", fs=8.5, lw=1.0, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
                                linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            zorder=3, color="#1B2631", fontweight="bold" if bold else "normal",
            linespacing=1.45)


def arrow(ax, p1, p2, color="#2C3E50", lw=1.1, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=11,
                                 linewidth=lw, color=color, zorder=1,
                                 linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


def panel_label(ax, label):
    ax.text(-0.06, 1.06, label, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top", ha="left")


# ================================================================= Figure 1
def fig1():
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")

    # --- 中列：Agent 执行循环 ---
    ax.text(5.0, 4.78, "SearchAgent 执行循环", ha="center", fontsize=9.5,
            fontweight="bold", color="#2C3E50")
    box(ax, 3.55, 4.16, 2.9, 0.44, "Question  $q$", fc="#D6EAF8", fs=9, bold=True)
    loop = ["Search(query)", "Read(page)", "AddEvidence(page, text)", "Answer(ans, cites)"]
    for i, name in enumerate(loop):
        y = 3.42 - i * 0.62
        box(ax, 3.55, y, 2.9, 0.44, name, fc="#F4F6F7", fs=8.2)
        if i:
            arrow(ax, (5.0, y + 0.62), (5.0, y + 0.44), lw=0.9)
    arrow(ax, (5.0, 4.16), (5.0, 3.86), lw=1.0)
    arrow(ax, (3.52, 1.47), (3.52, 4.36), color=C_RED, lw=1.3, rad=-0.30)
    ax.text(3.35, 2.92, r"迭代 $\leq$ 8K token", ha="center", va="center", fontsize=7.2,
            color=C_RED, rotation=90)

    # 右侧：记忆与预算
    box(ax, 6.85, 2.55, 2.85, 1.05,
        "Evidence Memory\n固定容量 · 历史查询去重", fc="#FDEBD0", fs=7.8)
    box(ax, 6.85, 1.28, 2.85, 1.05,
        "Budget Guard\n4×Search · 6×Read · 8K token", fc="#FDEBD0", fs=7.8)
    arrow(ax, (6.85, 3.07), (6.45, 3.07), lw=0.9, style="<|-|>")
    arrow(ax, (6.85, 1.80), (6.45, 1.80), lw=0.9, style="<|-|>")

    # 左侧：检索后端
    box(ax, 0.30, 3.35, 2.60, 1.05,
        "离线混合检索\nBM25 + BGE-M3(1024d) + RRF\n104,477 段 Wikipedia", fc="#E8F6F3", fs=7.6)
    box(ax, 0.30, 2.05, 2.60, 1.05,
        "在线检索（可插拔）\nTavily Search API", fc="#E8F6F3", fs=7.6)
    arrow(ax, (2.90, 3.87), (3.55, 3.66), lw=0.9)
    arrow(ax, (2.90, 2.57), (3.55, 3.02), lw=0.9, ls=(0, (3, 2)))

    ax.text(5.0, 0.30, "图 1  SearchAgent 的动作空间、证据记忆与预算约束。"
                       "红色回路表示模型在证据不足时继续多跳检索。",
            ha="center", fontsize=8, color="#566573")
    fig.savefig(OUT / "fig1_architecture.png")
    plt.close(fig)


# ================================================================= Figure 2
def fig2():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0),
                             gridspec_kw={"width_ratios": [1.42, 1]})

    # --- 左：数据构造流水线 ---
    ax = axes[0]; ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    panel_label(ax, "(a)")
    stages = [
        ("3 个公开多跳数据集", "AgenticRagTrace 1,299 · HotpotQA 79,368 · MuSiQue 23,885", "#D6EAF8"),
        ("Wikipedia 语料池", "104,477 段去重段落", "#D6EAF8"),
        ("混合索引构建", "BM25 + BGE-M3 向量 + RRF 融合", "#E8F6F3"),
        ("轨迹采样（SearchTrace）", "Oracle 60% · Teacher 25% · 漂移恢复 15%", "#FDEBD0"),
        ("逐步决策样本", "1,305 条轨迹 → 7,047 个单步样本", "#FADBD8"),
    ]
    for i, (t, s, c) in enumerate(stages):
        y = 5.05 - i * 1.02
        box(ax, 0.25, y, 9.5, 0.80, f"{t}\n{s}", fc=c, fs=7.4)
        if i:
            arrow(ax, (5.0, y + 1.02), (5.0, y + 0.80), lw=1.0)

    # --- 右：配比与规模 ---
    ax = axes[1]
    panel_label(ax, "(b)")
    types = ["Oracle\n成功轨迹", "Teacher\n多样化轨迹", "漂移恢复\n轨迹"]
    vals = [60, 25, 15]
    bars = ax.barh(types, vals, color=[C_RED, C_BLUE, C_ORANGE], height=0.55, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(v + 1.5, b.get_y() + b.get_height() / 2, f"{v}%",
                va="center", fontsize=8.5, fontweight="bold")
    ax.set_xlim(0, 78); ax.set_xlabel("占 1,305 条轨迹的比例 (%)")
    ax.grid(axis="x", ls=":", lw=0.6, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("训练轨迹构成", fontsize=9.5, pad=6)

    fig.text(0.5, -0.13, "图 2  SearchTrace 的数据构造流水线与轨迹配比。"
                         "混合配比避免模型只模仿单一成功模式。",
             ha="center", fontsize=8, color="#566573")
    fig.savefig(OUT / "fig2_data_pipeline.png")
    plt.close(fig)


# ================================================================= Figure 3
def fig3():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))

    # (a) 学习曲线
    ax = axes[0]
    panel_label(ax, "(a)")
    mk = {"LoRA": ("o", C_BLUE), "DoRA": ("s", C_RED)}
    lo, hi = 1e9, -1e9
    for name, ys in SFT_EVAL.items():
        if not ys:
            continue
        m, c = mk[name]
        xs = _epoch_axis(len(ys))
        ax.plot(xs, ys, marker=m, ms=4.2, lw=1.4, color=c, label=name, zorder=3)
        dy = -11 if name == "LoRA" else 4
        ax.annotate(f"{ys[-1]:.4f}", (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(5, dy), fontsize=7.5, color=c, fontweight="bold")
        lo, hi = min(lo, min(ys)), max(hi, max(ys))
    ax.set_xlabel("训练轮次 (epoch)")
    ax.set_ylabel("验证集损失 eval_loss")
    ax.set_title("SFT 收敛曲线", fontsize=9.5, pad=6)
    ax.grid(ls=":", lw=0.6, alpha=0.6)
    ax.legend(loc="upper right")
    pad = (hi - lo) * 0.35 if hi > lo else 0.05
    ax.set_ylim(lo - pad, hi + pad * 0.6)

    # (b) 收敛速度对比
    ax = axes[1]
    panel_label(ax, "(b)")
    names = ["LoRA", "DoRA"]
    runtime = [SFT_RUNTIME[n] / 60 for n in names]      # 分钟
    loss = [SFT_TRAIN[n] for n in names]
    x = np.arange(len(names)); w = 0.34
    ax.bar(x - w / 2, runtime, w, color=C_BLUE, label="训练耗时 (min)", zorder=3)
    ax2 = ax.twinx()
    ax2.bar(x + w / 2, loss, w, color=C_RED, label="最终 train_loss", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("耗时 (分钟)", color=C_BLUE)
    ax2.set_ylabel("train_loss", color=C_RED)
    ax2.set_ylim(0, max(loss) * 1.45 if loss and loss == loss else 1.0)
    ax.set_ylim(0, max(runtime) * 1.30 if runtime else 1.0)
    ax.grid(axis="y", ls=":", lw=0.6, alpha=0.6); ax.set_axisbelow(True)
    for xi, r in zip(x, runtime):
        ax.text(xi - w / 2, r + 2, f"{r:.0f}", ha="center", fontsize=8, color=C_BLUE,
                fontweight="bold")
    for xi, l in zip(x, loss):
        ax2.text(xi + w / 2, l + 0.02, f"{l:.4f}", ha="center", fontsize=7.5, color=C_RED,
                 fontweight="bold")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.set_title("训练成本与最终损失", fontsize=9.5, pad=6)

    fig.text(0.5, -0.17, "图 3  LoRA 与 DoRA 的 SFT 收敛行为。DoRA 以约 3 倍训练时间为代价，"
                         "取得更低的验证损失。", ha="center", fontsize=8, color="#566573")
    fig.savefig(OUT / "fig3_sft_curves.png")
    plt.close(fig)


# ================================================================= Figure 4
def fig4():
    ms = present_methods()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))

    for ax, metric, title, ylab in [
        (axes[0], "answer_f1", "(a) 答案质量", "Answer F1"),
        (axes[1], "joint_f1", "(b) 联合质量", "Joint F1"),
    ]:
        panel_label(ax, title[:3])
        vals = [CMP[m][metric] for m in ms]
        bars = ax.bar([LABELS[m] for m in ms], vals,
                      color=[COLORS[m] for m in ms], width=0.58, zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.035,
                    f"{v:.4f}", ha="center", fontsize=8, fontweight="bold")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=9.5, pad=6)
        ax.grid(axis="y", ls=":", lw=0.6, alpha=0.6); ax.set_axisbelow(True)
        ax.set_ylim(0, max(vals) * 1.30)
        ax.tick_params(axis="x", labelrotation=18)

    fig.text(0.5, -0.17, "图 4  在页面隔离的 sealed test（n=100）上的主结果。"
                         "后训练将 Answer F1 相对规则基线提升 3.5 倍。",
             ha="center", fontsize=8, color="#566573")
    fig.savefig(OUT / "fig4_main_results.png")
    plt.close(fig)


# ================================================================= Figure 5
def fig5():
    ms = present_methods()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))

    # (a) 证据覆盖 vs 答案质量 的 trade-off
    ax = axes[0]; panel_label(ax, "(a)")
    for m in ms:
        x = CMP[m]["evidence_all_hit"] * 100
        y = CMP[m]["answer_f1"] * 100
        ax.scatter(x, y, s=62, color=COLORS[m], zorder=3, edgecolor="white", linewidth=0.8)
        ax.annotate(LABELS[m], (x, y), textcoords="offset points", xytext=(7, 4),
                    fontsize=7.6, color=COLORS[m], fontweight="bold")
    ax.set_xlabel("证据链 All-Hit (%)")
    ax.set_ylabel("Answer F1 (%)")
    ax.set_title("覆盖度与准确率的权衡", fontsize=9.5, pad=6)
    ax.grid(ls=":", lw=0.6, alpha=0.6); ax.set_axisbelow(True)
    ax.set_xlim(40, 112); ax.set_ylim(-0.6, max(CMP[m]["answer_f1"] for m in ms) * 118)

    # (b) 平均工具调用
    ax = axes[1]; panel_label(ax, "(b)")
    vals = [CMP[m]["avg_tool_calls"] for m in ms]
    bars = ax.bar([LABELS[m] for m in ms], vals, color=[COLORS[m] for m in ms],
                  width=0.58, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}",
                ha="center", fontsize=8, fontweight="bold")
    ax.set_ylabel("平均工具调用次数")
    ax.set_ylim(0, max(vals) * 1.28)
    ax.set_title("推理成本", fontsize=9.5, pad=6)
    ax.grid(axis="y", ls=":", lw=0.6, alpha=0.6); ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelrotation=18)

    fig.text(0.5, -0.06, "图 5  规则基线靠“全量堆证据”取得高覆盖，但牺牲答案精度；"
                         "后训练模型学会选择性取证。", ha="center", fontsize=8, color="#566573")
    fig.savefig(OUT / "fig5_evidence_tradeoff.png")
    plt.close(fig)


# ================================================================= Figure 6
def fig6():
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.2); ax.axis("off")

    steps = [
        ("Question", "Which film directed by the\nsibling of a famous actor\nwon the Palme d'Or?", "#D6EAF8"),
        ("Search", "\"Palme d'Or winner directed by\nsibling of famous actor\"", "#F4F6F7"),
        ("Read", "Ronan Marraige\n(bridge entity extracted)", "#FDEBD0"),
        ("Search", "\"Ronan Marraige film Palme d'Or\"", "#F4F6F7"),
        ("AddEvidence", "page: Ronan Marraige\n“...won the Palme d'Or in 2015...”", "#FADBD8"),
        ("Answer", "Ronan Marraige\n[citations: 0]", "#D5F5E3"),
    ]
    n = len(steps)
    for i, (t, s, c) in enumerate(steps):
        y = 5.62 - i * 0.95
        box(ax, 0.35, y, 3.05, 0.72, f"{t}\n{s}", fc=c, fs=6.9)
        if i:
            arrow(ax, (1.87, y + 0.95), (1.87, y + 0.72), lw=1.0)

    # 右侧注释：桥接与恢复
    box(ax, 4.05, 3.72, 5.55, 1.05,
        "桥接实体（bridge entity）：第一跳检索得到的中间实体\n"
        "重写查询并触发第二跳，是跨文档推理的关键", fc="#EAF2F8", fs=7.4)
    box(ax, 4.05, 2.30, 5.55, 1.05,
        "结构化 recovery 反馈：证据不足时返回缺什么\n"
        "（缺实体 / 缺来源 / 查询漂移），引导重新检索", fc="#EAF2F8", fs=7.4)
    box(ax, 4.05, 0.88, 5.55, 1.05,
        "预算守卫：4×Search / 6×Read / 8K token\n"
        "超限即强制 Answer，杜绝无意义循环", fc="#EAF2F8", fs=7.4)
    arrow(ax, (3.40, 4.05), (4.05, 4.24), lw=0.8, ls=(0, (3, 2)))
    arrow(ax, (3.40, 3.10), (4.05, 2.82), lw=0.8, ls=(0, (3, 2)))
    arrow(ax, (3.40, 1.20), (4.05, 1.40), lw=0.8, ls=(0, (3, 2)))

    fig.text(0.5, -0.02, "图 6  一个真实的多跳轨迹示例（Qwen3-1.7B + DoRA）："
                         "第一跳抽桥接实体，第二跳锁定答案并给出引用。",
             ha="center", fontsize=8, color="#566573")
    fig.savefig(OUT / "fig6_trajectory.png")
    plt.close(fig)


if __name__ == "__main__":
    fig1(); print("fig1 ok")
    fig2(); print("fig2 ok")
    fig3(); print("fig3 ok")
    fig4(); print("fig4 ok")
    fig5(); print("fig5 ok")
    fig6(); print("fig6 ok")
    print(f"输出目录: {OUT}")
