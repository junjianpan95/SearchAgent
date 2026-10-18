"""生成 SearchAgent 项目推广图（一套 4 张，高清 PNG）。

图1：项目主视觉/封面
图2：系统架构图
图3：训练流程（SFT + 课程奖励 RL）
图4：实验效果对比
"""
from PIL import Image, ImageDraw, ImageFont

FONT_BOLD = "C:/Windows/Fonts/msyhbd.ttc"
FONT_REG = "C:/Windows/Fonts/msyh.ttc"
FONT_HEI = "C:/Windows/Fonts/simhei.ttf"

BG = (11, 16, 32)
BG2 = (26, 36, 64)
PANEL = (33, 45, 77)
BORDER = (44, 58, 94)
TEXT = (230, 236, 255)
MUTED = (139, 152, 184)
ACCENT = (79, 140, 255)
ACCENT2 = (124, 92, 255)
GREEN = (52, 211, 153)
ORANGE = (251, 191, 36)
CYAN = (34, 211, 238)
RED = (248, 113, 113)


def font(size, bold=True):
    path = FONT_BOLD if bold else FONT_REG
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.truetype(FONT_HEI, size)


def new_canvas(w, h):
    img = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(img)
    # 渐变背景
    for y in range(h):
        t = y / h
        c = tuple(int(BG[i] + (BG2[i] - BG[i]) * t) for i in range(3))
        d.line([(0, y), (w, y)], fill=c)
    return img, d


def round_rect(d, xy, r, fill, outline=None, width=1):
    d.rounded_rectangle(xy, radius=r, fill=fill, outline=outline, width=width)


def center_text(d, cx, y, text, f, fill=TEXT):
    w = d.textlength(text, font=f)
    d.text((cx - w / 2, y), text, font=f, fill=fill)


# ============================================================ 图1：封面
def make_cover():
    w, h = 1200, 675
    img, d = new_canvas(w, h)
    f_title = font(64, True)
    f_sub = font(28, False)
    f_badge = font(22, True)
    f_tag = font(24, True)

    center_text(d, w/2, 155, "SearchAgent", f_title, ACCENT)
    center_text(d, w/2, 240, "多跳检索 Agentic RAG 问答系统", f_sub, TEXT)

    # badge
    center_text(d, w/2, 305, "基于在线检索与多策略后训练", f_badge, ACCENT2)

    # 标签
    tags = ["Search / Read / AddEvidence / Answer", "BM25 + BGE-M3 + RRF", "Tavily 在线搜索",
            "LoRA/DoRA × GRPO/GSPO", "两阶段课程奖励"]
    y = 365
    for t in tags:
        tw = d.textlength(t, font=f_tag)
        x = w/2 - tw/2
        round_rect(d, (x-20, y, x+tw+20, y+46), 23, PANEL, BORDER, 2)
        center_text(d, w/2, y+8, t, f_tag, TEXT)
        y += 58

    center_text(d, w/2, 655, "让模型依据中间证据逐轮生成查询 · 选择支持句 · 输出带可核验引用的答案",
                font(20, False), MUTED)
    return img


# ============================================================ 图2：架构图
def make_arch():
    w, h = 1200, 800
    img, d = new_canvas(w, h)
    f_title = font(40, True)
    f_node = font(22, True)
    f_small = font(16, False)
    f_arrow = font(30, True)

    center_text(d, w/2, 40, "系统架构", f_title, TEXT)

    # 层1：问题集
    round_rect(d, (400, 100, 800, 160), 16, PANEL, BORDER, 2)
    center_text(d, 600, 112, "多跳问题集", f_node, TEXT)
    center_text(d, 600, 142, "SearchTrace 构造", f_small, MUTED)

    center_text(d, 600, 175, "↓", f_arrow, MUTED)

    # 层2：Agent
    round_rect(d, (350, 200, 850, 280), 16, (40, 60, 110), ACCENT, 2)
    center_text(d, 600, 212, "SearchAgent 核心", f_node, TEXT)
    center_text(d, 600, 248, "动作空间 · 证据记忆 · 预算约束 · 停止条件", f_small, MUTED)

    center_text(d, 600, 295, "↓ 四个结构化动作", f_small, MUTED)

    # 层3：动作
    actions = ["Search", "Read", "AddEvidence", "Answer"]
    colors = [ACCENT, CYAN, GREEN, ACCENT2]
    xs = [180, 420, 660, 900]
    for i, (a, c) in enumerate(zip(actions, colors)):
        round_rect(d, (xs[i]-100, 330, xs[i]+100, 410), 14, PANEL, c, 2)
        center_text(d, xs[i], 350, a, f_node, c)
        sub = ["生成查询", "读取页面", "选择支持句", "带引用回答"][i]
        center_text(d, xs[i], 380, sub, f_small, MUTED)

    center_text(d, 600, 425, "↓", f_arrow, MUTED)

    # 层4：检索后端
    round_rect(d, (250, 460, 610, 540), 16, PANEL, BORDER, 2)
    center_text(d, 430, 475, "本地混合索引", f_node, TEXT)
    center_text(d, 430, 505, "BM25 + BGE-M3 + RRF + Reranker", f_small, MUTED)

    round_rect(d, (640, 460, 1000, 540), 16, PANEL, BORDER, 2)
    center_text(d, 820, 475, "Tavily 在线搜索", f_node, ORANGE)
    center_text(d, 820, 505, "可插拔后端 · 消融对比", f_small, MUTED)

    center_text(d, 600, 565, "在线 vs 本地 消融", f_small, ACCENT2)

    # 层5：后训练
    round_rect(d, (200, 620, 1000, 720), 16, PANEL, ACCENT2, 2)
    center_text(d, 600, 640, "后训练：LoRA/DoRA SFT  →  GRPO/GSPO 课程奖励 RL", f_node, TEXT)
    center_text(d, 600, 678, "两阶段课程奖励 · KL 约束 · 零方差重采样 · reward 分布监控", f_small, MUTED)

    return img


# ============================================================ 图3：训练流程
def make_train():
    w, h = 1200, 800
    img, d = new_canvas(w, h)
    f_title = font(40, True)
    f_node = font(20, True)
    f_small = font(15, False)

    center_text(d, w/2, 40, "训练方案", f_title, TEXT)

    # 阶段1 SFT
    round_rect(d, (80, 110, 560, 300), 16, PANEL, ACCENT, 2)
    center_text(d, 320, 125, "Stage 1 · SFT", font(26, True), ACCENT)
    center_text(d, 320, 170, "LLaMA-Factory + DeepSpeed", f_node, TEXT)
    center_text(d, 320, 200, "LoRA / DoRA", f_node, TEXT)
    center_text(d, 320, 240, "仅监督 Agent 生成 token", f_small, MUTED)
    center_text(d, 320, 262, "查询 · 动作 · 证据 · 答案", f_small, MUTED)
    center_text(d, 320, 290, "Joint F1 + 证据链 + 工具成本选 checkpoint", f_small, GREEN)

    # 箭头
    center_text(d, 600, 200, "→", font(50, True), MUTED)

    # 阶段2 RL
    round_rect(d, (640, 110, 1120, 300), 16, PANEL, ACCENT2, 2)
    center_text(d, 880, 125, "Stage 2 · RL", font(26, True), ACCENT2)
    center_text(d, 880, 170, "verl + vLLM + Ray + FSDP", f_node, TEXT)
    center_text(d, 880, 200, "GRPO / GSPO  on-policy", f_node, TEXT)
    center_text(d, 880, 240, "真实检索环境 · 每题采样 4 条轨迹", f_small, MUTED)
    center_text(d, 880, 262, "两阶段课程奖励", f_small, MUTED)
    center_text(d, 880, 290, "KL 约束 + 零方差重采样", f_small, GREEN)

    # 课程奖励
    center_text(d, w/2, 440, "课程奖励设计", font(28, True), TEXT)

    round_rect(d, (80, 480, 560, 620), 16, PANEL, GREEN, 2)
    center_text(d, 320, 495, "Stage 1 优化", font(22, True), GREEN)
    center_text(d, 320, 540, "新增 supporting-fact", f_node, TEXT)
    center_text(d, 320, 570, "引用有效性 · grounding", f_node, TEXT)

    round_rect(d, (640, 480, 1120, 620), 16, PANEL, ORANGE, 2)
    center_text(d, 880, 495, "Stage 2 强化", font(22, True), ORANGE)
    center_text(d, 880, 540, "Answer EM / F1", f_node, TEXT)
    center_text(d, 880, 570, "保留证据约束", f_node, TEXT)

    # 惩罚
    center_text(d, w/2, 660, "惩罚项", font(22, True), RED)
    pen = ["重复查询", "冗余证据", "非法动作", "工具成本"]
    xs = [240, 480, 720, 960]
    for p, x in zip(pen, xs):
        round_rect(d, (x-90, 690, x+90, 730), 14, PANEL, RED, 2)
        center_text(d, x, 702, p, f_node, TEXT)

    return img


# ============================================================ 图4：实验效果
def make_result():
    w, h = 1200, 800
    img, d = new_canvas(w, h)
    f_title = font(40, True)
    f_num = font(44, True)
    f_label = font(18, False)
    f_small = font(15, False)

    center_text(d, w/2, 40, "实验效果", f_title, TEXT)

    # 指标卡片
    metrics = [
        ("Answer F1", "0.43 → 0.53", "+23%", GREEN),
        ("Joint F1", "+4%", "GRPO→GSPO", ACCENT),
        ("证据链 All-Hit", "+6%", "完整召回", ACCENT2),
        ("冗余调用率", "28% → 22%", "-6pt", CYAN),
        ("工具调用", "4.8 → 4.2", "-0.6", ORANGE),
    ]
    x0 = 90
    cw = 204
    for i, (name, val, delta, color) in enumerate(metrics):
        x = x0 + i * (cw + 6)
        round_rect(d, (x, 110, x+cw, 300), 14, PANEL, BORDER, 2)
        center_text(d, x+cw/2, 130, name, f_label, MUTED)
        center_text(d, x+cw/2, 185, val.split("→")[0], f_num, TEXT)
        if "→" in val:
            center_text(d, x+cw/2, 240, "→ " + val.split("→")[1], font(30, True), color)
        center_text(d, x+cw/2, 272, delta, f_small, color)

    # 提升路径表
    center_text(d, w/2, 340, "Base ReAct → SFT → GRPO/GSPO 提升路径", font(26, True), TEXT)

    rows = [
        ("指标", "Base ReAct", "SFT", "GRPO/GSPO"),
        ("Answer F1", "0.43", "0.50", "0.53"),
        ("Joint F1", "-", "-", "+4%"),
        ("证据链 All-Hit", "-", "-", "+6%"),
        ("冗余调用率", "28%", "-", "22%"),
        ("平均工具调用", "4.8", "-", "4.2"),
    ]
    ty = 390
    rh = 55
    for ri, row in enumerate(rows):
        y = ty + ri * rh
        for ci, cell in enumerate(row):
            cx = 200 + ci * 270
            color = TEXT
            if ri == 0:
                color = MUTED
            elif ci == 3 and ri > 1:
                color = GREEN
            center_text(d, cx, y, cell, f_label if ri == 0 else font(20, True), color)
        if ri < len(rows) - 1:
            d.line([(120, y+rh-5), (1080, y+rh-5)], fill=BORDER)

    center_text(d, w/2, 740, "MuSiQue zero-shot 保持约 +1.5% 增益", font(20, True), ORANGE)

    return img


def main():
    import os
    out_dir = "docs/images"
    os.makedirs(out_dir, exist_ok=True)
    images = {
        "cover": make_cover(),
        "arch": make_arch(),
        "train": make_train(),
        "result": make_result(),
    }
    for name, img in images.items():
        path = f"{out_dir}/{name}.png"
        img.save(path, quality=95)
        print(f"[IMG] 生成 {path} ({img.size[0]}x{img.size[1]})")


if __name__ == "__main__":
    main()
