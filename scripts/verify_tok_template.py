#!/usr/bin/env python3
"""验证合并模型的 chat_template 是否可用，并打印渲染结果（含 enable_thinking 两种开关）。"""
from transformers import AutoTokenizer

P = "/home/pan/SearchAgent/data/models/qwen3_1.7b_sft_merged"
tok = AutoTokenizer.from_pretrained(P)
msgs = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Question: x\nNext action:"},
]

print("chat_template 存在:", tok.chat_template is not None)

for kw in [{}, {"enable_thinking": False}]:
    try:
        out = tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False, **kw
        )
        print(f"\n--- kwargs={kw} ---")
        print(repr(out))
    except Exception as e:
        print(f"kwargs={kw} 失败: {type(e).__name__}: {e}")
