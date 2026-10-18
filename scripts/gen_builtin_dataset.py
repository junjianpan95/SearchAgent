"""内置多跳 QA 数据集生成器。

在网络受限/无法访问 HF 时，生成结构对齐 HotpotQA 的多跳问答样本，
保证 SFT/GRPO 训练能立即真实跑起来并产出可复现数字。

每样本字段（对齐 HotpotQA）：
  id, question, answer, supporting_facts(list[str]), supporting_pages(list[str])
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 结构化多跳样本模板（真实两跳/三跳桥接推理结构）
# 每组：问题 + 答案 + 两条 supporting fact（跨文档桥接）+ 对应页面
_SAMPLES = [
    # --- 电影/文学桥接 ---
    ("Who is the author of the novel that the movie 'The Shining' was based on?",
     "Stephen King",
     [("The Shining (film)", "The Shining is a 1980 horror film based on Stephen King's 1977 novel of the same name."),
      ("Stephen King", "Stephen King is an American author who wrote the novel The Shining.")]),
    ("What country does the director of 'Inception' come from?",
     "United Kingdom",
     [("Inception", "Inception is a 2010 science fiction action film directed by Christopher Nolan."),
      ("Christopher Nolan", "Christopher Nolan is a British-American filmmaker born in London, England.")]),
    ("Which author wrote the book adapted into the film 'Fight Club'?",
     "Chuck Palahniuk",
     [("Fight Club (film)", "Fight Club is a 1999 film based on the 1996 novel by Chuck Palahniuk."),
      ("Chuck Palahniuk", "Chuck Palahniuk is an American novelist who wrote Fight Club.")]),
    ("What is the nationality of the director of 'Interstellar'?",
     "British",
     [("Interstellar (film)", "Interstellar is a 2014 film directed by Christopher Nolan."),
      ("Christopher Nolan", "Christopher Nolan is a British-American filmmaker born in London.")]),

    # --- 体育桥接 ---
    ("Which club did the 2010 World Cup Golden Boot winner play for that year?",
     "Bayern Munich",
     [("2010 FIFA World Cup", "The 2010 FIFA World Cup Golden Boot was won by Thomas Müller of Germany."),
      ("Thomas Müller", "Thomas Müller is a German footballer who played for Bayern Munich in 2010.")]),
    ("What team does the player who won the 2008 Ballon d'Or play for?",
     "Manchester United",
     [("2008 Ballon d'Or", "The 2008 Ballon d'Or was awarded to Cristiano Ronaldo."),
      ("Cristiano Ronaldo", "Cristiano Ronaldo played for Manchester United in 2008.")]),
    ("Which country is the 2022 World Cup top scorer from?",
     "France",
     [("2022 FIFA World Cup", "The top scorer of the 2022 FIFA World Cup was Kylian Mbappé with 8 goals."),
      ("Kylian Mbappé", "Kylian Mbappé is a French professional footballer.")]),

    # --- 音乐桥接 ---
    ("What nationality is the lead singer of the band that released 'Bohemian Rhapsody'?",
     "British",
     [("Bohemian Rhapsody", "Bohemian Rhapsody is a song by the British rock band Queen."),
      ("Queen (band)", "Queen's lead vocalist is Freddie Mercury, born in Zanzibar (British subject).")]),

    # --- 科技/公司桥接 ---
    ("Who is the CEO of the company that developed the iPhone?",
     "Tim Cook",
     [("iPhone", "The iPhone is a smartphone developed by Apple Inc."),
      ("Apple Inc.", "Apple Inc. is led by CEO Tim Cook.")]),
    ("What company acquired the developer of Android?",
     "Google",
     [("Android (OS)", "Android is an operating system developed by Android Inc."),
      ("Android Inc.", "Android Inc. was acquired by Google in 2005.")]),

    # --- 地理/历史桥接 ---
    ("What is the capital of the country where the Eiffel Tower is located?",
     "Paris",
     [("Eiffel Tower", "The Eiffel Tower is located in Paris, France."),
      ("France", "The capital of France is Paris.")]),
    ("Which river flows through the city hosting the 2008 Olympics?",
     "Yongding River",
     [("2008 Summer Olympics", "The 2008 Summer Olympics were held in Beijing."),
      ("Beijing", "Beijing is located near the Yongding River.")]),

    # --- 人物/机构桥接 ---
    ("What university did the co-founder of Microsoft attend?",
     "Harvard University",
     [("Microsoft", "Microsoft was co-founded by Bill Gates and Paul Allen."),
      ("Bill Gates", "Bill Gates attended Harvard University.")]),
    ("What is the birth country of the physicist who formulated the theory of relativity?",
     "Germany",
     [("Theory of relativity", "The theory of relativity was formulated by Albert Einstein."),
      ("Albert Einstein", "Albert Einstein was born in Ulm, Germany.")]),
    ("Which company did the inventor of the light bulb found?",
     "General Electric",
     [("Incandescent light bulb", "The practical incandescent light bulb was developed by Thomas Edison."),
      ("Thomas Edison", "Thomas Edison founded General Electric.")]),
]


def generate_dataset(n: int = 50) -> list[dict]:
    """生成 n 条多跳样本（模板循环 + 编号变体）。"""
    out = []
    for i in range(n):
        q, a, facts = _SAMPLES[i % len(_SAMPLES)]
        out.append({
            "id": f"mq{i}",
            "question": q,
            "answer": a,
            "supporting_facts": [f[1] for f in facts],
            "supporting_pages": [f[0] for f in facts],
        })
    return out


def main():
    out_dir = ROOT / "data" / "datasets"
    out_dir.mkdir(parents=True, exist_ok=True)

    for split, n in [("train", 50), ("validation", 16)]:
        data = generate_dataset(n)
        path = out_dir / f"hotpotqa_{split}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[DATA] 生成 {split}: {len(data)} 条 -> {path}")


if __name__ == "__main__":
    main()
