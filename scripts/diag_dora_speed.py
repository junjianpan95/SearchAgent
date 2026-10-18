"""诊断 lora vs dora 的 agent 运行行为差异（轮次数 + 耗时）。"""
import sys, time, json
from pathlib import Path

ROOT = Path("E:/Life/1/SearchAgent")
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.retrieval.index_loader import CachedHybridRetriever
from src.agent import SearchAgent

def load_cfg():
    from src.config import load_config
    return load_config(str(ROOT / "configs" / "local_smoke.yaml"))

def make_llm(adapter_dir):
    base = ROOT / "data" / "models" / "Qwen3-1.7B"
    tok = AutoTokenizer.from_pretrained(str(base), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(base), torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model = model.merge_and_unload()
    model.eval()
    # 复用 run_compare 的解析
    from scripts.run_compare import parse_text_action
    def llm_fn(context=None, state=None, **kw):
        question = context[0].replace("Question: ", "") if context else ""
        lines = [f"Question: {question}"]
        if state and state.memory and len(state.memory) > 0:
            lines.append("Evidence:")
            for e in state.memory.items:
                lines.append(f"- {e.text[:100]}")
        if state and state.trajectory:
            prev = state.trajectory[-1]
            if prev.get("action", {}).get("type") == "Search":
                hits = (prev.get("result") or {}).get("hits", [])
                if hits:
                    lines.append("Search results:")
                    for h in hits[:3]:
                        lines.append(f"- {h.get('page','')}: {h.get('text','')[:100]}")
        lines.append("Next action:")
        instruction = "\n".join(lines)
        messages = [{"role":"system","content":"You are a helpful assistant."},
                    {"role":"user","content":instruction}]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=32, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        return parse_text_action(text)
    return llm_fn

def main():
    from scripts.run_compare import load_sealed_test
    cfg = load_cfg()
    data = load_sealed_test(5)  # 只取 5 条
    retriever = CachedHybridRetriever(device="cpu")

    for method in ["lora", "dora"]:
        adapter = ROOT / "data" / "models" / f"sft_{method}"
        llm = make_llm(adapter)
        print(f"\n===== {method} =====")
        t0 = time.time()
        for i, item in enumerate(data):
            ts = time.time()
            agent = SearchAgent(
                retriever=retriever, llm_fn=llm,
                max_search=cfg.agent["max_search"], max_read=cfg.agent["max_read"],
                max_evidence=cfg.agent["max_evidence"], token_budget=cfg.agent["token_budget"],
                dedup_queries=cfg.agent["dedup_queries"],
                bridge_entity_direct_read=cfg.agent["bridge_entity_direct_read"],
                recovery_feedback=cfg.agent["recovery_feedback"],
            )
            state = agent.run(item["question"])
            n_tool = state.budget.snapshot()["tool_calls"]
            print(f"  样本{i}: tool_calls={n_tool}  steps={len(state.trajectory)}  {time.time()-ts:.1f}s")
        print(f"  {method} 总计 {time.time()-t0:.1f}s (5条)")

if __name__ == "__main__":
    main()
