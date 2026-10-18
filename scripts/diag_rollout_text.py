#!/usr/bin/env python3
"""诊断 RL 起点策略（SFT-LoRA 合并模型）的实际生成输出。

验证三件事：
1. Qwen3 在 add_generation_prompt 后是否输出 <|...thinking...> 前缀（污染动作解析）
2. 模型是否能在多轮里正常输出 Search:/Answer: 动作
3. 对比 run_compare.py 的 parse_text_action 与 RL 的 parse_first_action 行为差异
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/home/pan/SearchAgent/data/models/qwen3_1.7b_sft_merged"
QUESTION = "Which RLL encoding stores more data bits per inch, (1,3) RLL encoding or (2,7) RLL encoding?"


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Question: {QUESTION}\nNext action:"},
    ]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    print("===== 渲染后的 prompt 尾部 =====")
    print(repr(prompt[-120:]))
    print()

    inputs = tok(prompt, return_tensors="pt").to(model.device)
    for temp, do_sample in [(1.0, True), (0.0, False)]:
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=do_sample,
                temperature=temp,
                top_p=0.95,
                pad_token_id=tok.eos_token_id,
            )
        text = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"===== 生成 (temp={temp}, sample={do_sample}) =====")
        print(repr(text))
        print()

    # 对比两种解析器
    from scripts.run_compare import parse_text_action as eval_parse
    from src.train.searchagent_loop import parse_first_action as rl_parse

    print("===== 解析器对比（用贪心结果）=====")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print("eval parse_text_action ->", eval_parse(text))
    print("rl   parse_first_action ->", rl_parse(text))


if __name__ == "__main__":
    main()
