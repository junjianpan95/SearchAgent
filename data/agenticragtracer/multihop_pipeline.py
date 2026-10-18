"""
多跳（multi-hop）QA 数据集自动生成流水线。

LLM 后端：NVIDIA NIM 免费端点，默认模型 GLM-5.3-Flash。
所有配置走环境变量，不需要改代码：

  NVIDIA_API_KEY   必填。build.nvidia.com 生成的 nvapi- 开头 Key
  NIM_MODEL        默认 z-ai/glm-5.3-flash
  NIM_BASE_URL     默认 https://integrate.api.nvidia.com/v1
  NIM_RPM          每分钟请求上限，免费层约 40（默认 40）
  NIM_CONCURRENCY  同时在飞的请求数上限（默认 8）
  NIM_MAX_TOKENS   单次最大输出。GLM 会先思考再回答，别设太小（默认 8192）
  NIM_TIMEOUT      单请求超时秒数（默认 600）
  NIM_MAX_RETRIES  429/5xx 重试次数（默认 6）
  NIM_TEMPERATURE  可选，不设则用服务端默认
  NIM_EXTRA_BODY   可选，JSON 字符串，原样并入请求体
                   例：{"chat_template_kwargs": {"enable_thinking": false}}

Windows 设置示例：
  set NVIDIA_API_KEY=nvapi-xxxxxxxx
  python multihop_run.py
"""

import json
import os
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Literal, Any

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
import time
import yaml
import re
import string
from collections import Counter
import random
from itertools import combinations
import tiktoken

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_yaml(file: str):
    with open(file, 'r', encoding="utf-8") as f:
        return yaml.safe_load(f)

# ======================= NVIDIA NIM 配置 =======================
NIM_BASE_URL = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/")
API_URL = os.getenv("NIM_API_URL", f"{NIM_BASE_URL}/chat/completions")
API_KEY = (os.getenv("NVIDIA_API_KEY") or os.getenv("NIM_API_KEY") or "").strip()
DEFAULT_MODEL = os.getenv("NIM_MODEL", "z-ai/glm-5.3-flash")

MAX_TOKENS = int(os.getenv("NIM_MAX_TOKENS", "8192"))
REQUEST_TIMEOUT = int(os.getenv("NIM_TIMEOUT", "600"))
MAX_RETRIES = int(os.getenv("NIM_MAX_RETRIES", "6"))
RPM_LIMIT = int(os.getenv("NIM_RPM", "40"))
MAX_CONCURRENCY = int(os.getenv("NIM_CONCURRENCY", "8"))

_temp = os.getenv("NIM_TEMPERATURE", "").strip()
TEMPERATURE = float(_temp) if _temp else None

_extra = os.getenv("NIM_EXTRA_BODY", "").strip()
try:
    EXTRA_BODY = json.loads(_extra) if _extra else {}
except json.JSONDecodeError as e:
    raise RuntimeError(f"NIM_EXTRA_BODY 不是合法 JSON: {e}") from e


class _NimRateLimiter:
    """均匀节流：任意 60s 窗口内请求数不超过 rpm，从根上避免打满免费层被 429。"""

    def __init__(self, rpm: int):
        self.rpm = max(1, int(rpm))
        self._interval = 60.0 / self.rpm
        self._lock = threading.Lock()
        self._next_slot = 0.0
        self._total = 0
        self._start = time.monotonic()

    def acquire(self):
        with self._lock:
            now = time.monotonic()
            wait = self._next_slot - now
            if wait < 0:
                wait = 0.0
            self._next_slot = max(now, self._next_slot) + self._interval
            self._total += 1
            total = self._total
        if wait > 0:
            time.sleep(wait)
        if total % 100 == 0:
            elapsed = max(time.monotonic() - self._start, 1e-6)
            logging.info("[NIM] 累计请求 %d 次，实测 %.1f req/min", total, total / elapsed * 60)


_LIMITER = _NimRateLimiter(RPM_LIMIT)
_SEMAPHORE = threading.BoundedSemaphore(max(1, MAX_CONCURRENCY))
_SESSION = requests.Session()
_SESSION.mount(
    "https://",
    HTTPAdapter(pool_connections=32, pool_maxsize=max(32, MAX_CONCURRENCY * 2), max_retries=0),
)


def _extract_message_text(data: dict) -> str:
    """GLM 是思考型模型：content 可能为空而正文在 reasoning_content 里，这里统一兜住。"""
    choices = data.get("choices") or []
    if not choices:
        raise ValueError(f"响应中没有 choices: {str(data)[:300]}")

    choice = choices[0]
    msg = choice.get("message") or {}
    content = msg.get("content")

    if isinstance(content, list):  # 兼容分块 content
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))

    if isinstance(content, str) and content.strip():
        return content

    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    if isinstance(reasoning, list):
        reasoning = "".join(p.get("text", "") for p in reasoning if isinstance(p, dict))

    if choice.get("finish_reason") == "length":
        raise ValueError(
            "输出被 max_tokens 截断且 content 为空（思考过程吃满了预算）；"
            "请调大 NIM_MAX_TOKENS（如 16384）"
        )
    if reasoning and reasoning.strip():
        logging.warning("[NIM] content 为空，回退使用 reasoning_content")
        return reasoning

    raise ValueError(f"响应 content 为空: {str(data)[:300]}")


def _openai_chat_api(messages, model=None, timeout=None, **kwargs):
    """调用 NIM 的 OpenAI 兼容 /chat/completions，带全局限流 + 并发闸 + 指数退避。"""
    if not API_KEY:
        raise RuntimeError(
            "未设置 NVIDIA_API_KEY。去 build.nvidia.com 生成 nvapi- 开头的 Key，"
            "然后 set NVIDIA_API_KEY=nvapi-xxxx 再运行。"
        )

    model = model or DEFAULT_MODEL
    timeout = timeout or REQUEST_TIMEOUT

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
    }
    if TEMPERATURE is not None:
        payload["temperature"] = TEMPERATURE
    if EXTRA_BODY:
        payload.update(EXTRA_BODY)
    payload.update(kwargs)

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        _LIMITER.acquire()
        try:
            with _SEMAPHORE:
                resp = _SESSION.post(API_URL, headers=headers, json=payload, timeout=timeout)
        except requests.RequestException as e:
            last_err = e
            delay = min(2 ** attempt, 60) + random.uniform(0, 1.5)
            logging.warning("[NIM] 网络异常 %d/%d：%s，%.1fs 后重试", attempt, MAX_RETRIES, e, delay)
            time.sleep(delay)
            continue

        if resp.status_code == 200:
            try:
                return _extract_message_text(resp.json())
            except Exception as e:
                last_err = e
                logging.warning("[NIM] 响应解析失败 %d/%d：%s", attempt, MAX_RETRIES, e)
                time.sleep(min(2 ** attempt, 30))
                continue

        body = resp.text[:500]

        if resp.status_code in (401, 403):
            raise RuntimeError(f"NVIDIA NIM 鉴权失败 {resp.status_code}，检查 NVIDIA_API_KEY 是否有效：{body}")
        if resp.status_code == 402:
            raise RuntimeError(f"NVIDIA NIM 额度/账单问题 402：{body}")
        if resp.status_code == 404:
            raise RuntimeError(
                f"NVIDIA NIM 模型不存在 404：model={model}。免费模型清单一周一变，"
                f"去 build.nvidia.com 核对 id：{body}"
            )

        retryable = resp.status_code == 429 or 500 <= resp.status_code < 600
        if retryable and attempt < MAX_RETRIES:
            retry_after = resp.headers.get("Retry-After", "")
            try:
                delay = float(retry_after) if retry_after else min(2 ** attempt, 60)
            except ValueError:
                delay = min(2 ** attempt, 60)
            delay += random.uniform(0, 1.5)
            logging.warning(
                "[NIM] HTTP %s %d/%d，%.1fs 后重试｜%s",
                resp.status_code, attempt, MAX_RETRIES, delay, body,
            )
            time.sleep(delay)
            continue

        raise RuntimeError(f"API Error {resp.status_code}: {body}")

    raise RuntimeError(f"NIM 请求失败（已重试 {MAX_RETRIES} 次）：{last_err}")

def _clean_json_block(item: str) -> str:
    text = (item or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^\s*```(?:json|JSON)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def _parse_json_response(resp_text: str):
    """先按整体解析；失败再扫全文抓第一个能解出来的 JSON 值（模型偶发夹带解释文字）。"""
    text = _clean_json_block(resp_text)
    try:
        return json.loads(text)
    except Exception:
        pass

    decoder = json.JSONDecoder()
    for m in re.finditer(r"[\[{]", text):
        start = m.start()
        try:
            value, _ = decoder.raw_decode(text[start:])
        except Exception:
            continue
        if isinstance(value, dict):
            return [value]   # 与原实现一致：裸对象包成单元素列表
        if isinstance(value, list):
            return value

    raise ValueError(f"Failed to parse JSON:\n{resp_text[:1200]}")


def run_llm_prompt(prompt, developer_prompt=None, model=None, return_json=False, max_retries=3):
    model = model or DEFAULT_MODEL
    messages = []
    if developer_prompt:
        messages.append({"role": "system", "content": developer_prompt})
    messages.append({"role": "user", "content": prompt})

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            resp_text = _openai_chat_api(messages, model=model)
            if return_json:
                return _parse_json_response(resp_text)
            return resp_text
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                delay = min(2 ** attempt, 20) + random.uniform(0, 1)
                logging.warning(
                    "[LLM] 第 %d/%d 次调用失败：%s，%.1fs 后重试",
                    attempt, max_retries, e, delay,
                )
                time.sleep(delay)
    raise RuntimeError(f"run_llm_prompt failed: {last_error}")

# ========= llm judge =======
def llm_judge(question, golden_answer, other_answer, llm_judge_prompt, num_parallel_predictions, model=DEFAULT_MODEL):
    prompt = f"""Input:\nQuestion: {question}\nGolden answer: {golden_answer}\nOther answer: {other_answer}"""

    def _judge_once():
        result = run_llm_prompt(prompt, developer_prompt=llm_judge_prompt, model=model, return_json=True)
        if result is None:
            return None
        return {
            "answer_score": result.get("answer_score", 0),
            "answer_reason": result.get("answer_reason", "")
        }

    results = []
    with ThreadPoolExecutor(max_workers=num_parallel_predictions) as exe:
        for r in exe.map(lambda _: _judge_once(), range(num_parallel_predictions)):
            if r is not None:
                results.append(r)

    if not results:
        return {
            "avg_score": 0,
            "reasons": [],
            "raw_scores": []
        }

    avg_score = sum(r["answer_score"] for r in results) / len(results)
    reasons = [r["answer_reason"] for r in results]
    raw_scores = [r["answer_score"] for r in results]

    return {
        "avg_score": avg_score,
        "reasons": reasons,
        "raw_scores": raw_scores
    }

# ========= F1 score =========
def normalize_answer(s: str) -> str:
    if s.strip() in ["A", "B", "C", "D", "E"]:
        return s.strip().upper()

    def remove_articles(text):
        return re.sub(r"\b(a|an|the|do|does|is|are|was|were|of|under|in|at|on|with|by|for|from|about)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction: str, ground_truths) -> float:
    if prediction is None or ground_truths is None:
        return 0.0

    if prediction.startswith("I cannot answer this question"):
        return 0.0

    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]

    max_f1 = 0.0

    for ground_truth in ground_truths:
        if ground_truth is None:
            continue

        normalized_prediction = normalize_answer(prediction)
        normalized_ground_truth = normalize_answer(ground_truth)

        if normalized_prediction in ["yes", "no", "noanswer"] or normalized_ground_truth in ["yes", "no", "noanswer"]:
            if normalized_prediction != normalized_ground_truth:
                continue

        pred_tokens = normalized_prediction.split()
        gold_tokens = normalized_ground_truth.split()
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            continue

        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        max_f1 = max(max_f1, f1)

    return max_f1

def _tokens(text: str) -> List[str]:
    return re.findall(r'\w+', text, flags=re.UNICODE)

def _years(text):
    return re.findall(r'\b\d{4}s?\b', text, flags=re.UNICODE | re.IGNORECASE)

def is_numeric(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def match_year_title(text):
    pattern = r'^"?(\d+s?|\d{4}s)"?\n(.*)'
    match = re.match(pattern, text)
    if match:
        return match.group(1)
    return None

def filtertokens(tokens):
    filter_tokens = []
    prepositions = ["Of", "Under", "In", "At", "On", "With", "By", "For", "From", "About", "An", "The", "Do", "Does", "Is", "Were", "Was", "Are"]
    for token in tokens:
        if (token[0].isupper() or token.isupper() or is_numeric(token)) and token not in prepositions:
            filter_tokens.append(token)
    return filter_tokens

def simple_partial_presence(phrase: str, sentence: str) -> bool:
    p_tokens = _tokens(phrase)
    p_tokens = filtertokens(p_tokens)
    s_tokens = _tokens(sentence)
    s_tokens = filtertokens(s_tokens)

    plen = len(p_tokens)
    for i in range(len(s_tokens) - plen + 1):
        if s_tokens[i:i+plen] == p_tokens:
            #print("full in")
            return False
        
    p_set = set(p_tokens)
    s_set = set(s_tokens)
    if len(p_set & s_set) > 0:
        #print("partial in")
        return True
    else:
        #print("not in")
        return True

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = a.intersection(b)
    union = a.union(b)
    return len(inter) / len(union)

def _get_title(doc: str):
    title = doc.split('\n')[0].strip('"')
    return title

# ========= MultiHopPipeline =========
class MultiHopPipeline:
    def __init__(
        self,
        model_id,
        retriever_url,
        prompt_file,
        benchmark_base_dir: str= "your path to benchmarks",
        token_threshold: float = 0.8
    ):
        self.model_id = model_id
        self.retriever_url = retriever_url
        self.prompts = load_yaml(prompt_file)

        self.benchmark_base_dir = benchmark_base_dir
        self.token_threshold = token_threshold

        self._load_benchmarks()

    def _load_benchmarks(self):
        base_dir = self.benchmark_base_dir
        musique_path = os.path.join(base_dir, "musique/dev.jsonl")
        wiki2_path = os.path.join(base_dir, "2wikimultihopqa/dev.jsonl")
        hotpot_path = os.path.join(base_dir, "hotpotqa/dev.jsonl")

        self.benchmark_norm_set = set()
        self.benchmark_norm_to_orig = {}
        self.benchmark_token_sets = {}   # norm -> frozenset(tokens)
        self.benchmark_inv = {}         # token -> set(norm)

        def add_title(raw_title):
            norm = normalize_answer(raw_title)
            if not norm:
                return
            if norm not in self.benchmark_norm_set:
                self.benchmark_norm_set.add(norm)
                # keep first-seen original
                self.benchmark_norm_to_orig[norm] = raw_title
            toks = frozenset(_tokens(norm))
            if toks:
                self.benchmark_token_sets[norm] = toks
                for t in toks:
                    self.benchmark_inv.setdefault(t, set()).add(norm)

        # load musique
        if os.path.exists(musique_path):
            with open(musique_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        supporting_list = data.get("metadata", {}).get("question_decomposition", [])
                        for supporting in supporting_list:
                            title = supporting.get("support_paragraph", {}).get("title", "")
                            if title:
                                add_title(title)
                    except Exception:
                        continue

        # load 2wiki
        if os.path.exists(wiki2_path):
            with open(wiki2_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        supporting_titles = data.get("metadata", {}).get("supporting_facts", {}).get("title", [])
                        for title in supporting_titles:
                            if title:
                                add_title(title)
                    except Exception:
                        continue

        # load hotpot
        if os.path.exists(hotpot_path):
            with open(hotpot_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        supporting_titles = data.get("metadata", {}).get("supporting_facts", {}).get("title", [])
                        for title in supporting_titles:
                            if title:
                                add_title(title)
                    except Exception:
                        continue

        print(f"[benchmarks] loaded {len(self.benchmark_norm_set)} normalized titles; token index size: {len(self.benchmark_inv)}")

    def retrieve_docs(self, query: str, original_docs: List[str], topk: int, num_hop: int) -> List[str]:
        """
        【Step 1: 用答案检索新文档】
        
        参数:
            query: 上一跳的答案（如 "Cheryl B. Schrader"）
            original_docs: 已用过的文档列表（避免重复）
            topk: 返回的文档数量（如 10）
            num_hop: 当前跳数（用于调整检索数量）
        
        返回:
            List[str]: 过滤后的新文档列表
        
        流程:
            1. 调用 Retriever 服务（E5 + FAISS）检索 15+num_hop 个文档
            2. 过滤：排除已用文档 + benchmark 重复文档
            3. 返回 topk 个最相关的文档
        """
        retri_num = random.randint(5, 15)  # 随机检索 5-15 个额外文档
        response = requests.post(
            self.retriever_url,  # http://localhost:8000/retrieve
            json={"query": query, "topk": retri_num + num_hop},  # 如 query="Cheryl B. Schrader", topk=16
            timeout=1200
        )
        data = response.json()
        all_docs = [doc["contents"] for doc in data.get("results", [])]  # 提取文档内容

        unique_docs = []
        for doc in all_docs:
            is_vaild = True
            for original_doc in original_docs:
                if doc.strip() == original_doc.strip():
                    is_vaild = False
                    break
            if not is_vaild:
                continue

            title = _get_title(doc)
            norm_title = normalize_answer(title)

            is_bench_match = False
            if norm_title in self.benchmark_norm_set:
                is_bench_match = True
                
            if is_bench_match:
                continue

            if doc not in unique_docs:
                unique_docs.append(doc)

        filter_docs = []
        for doc in unique_docs:
            if "(number)" in doc or "(decade)" in doc or len(_years(doc)) > 20 or match_year_title(doc):
                continue
            filter_docs.append(doc)

        if len(filter_docs) > topk:
            filter_docs = filter_docs[-topk:]
        return filter_docs

    def compare_verify(self, prompt: str, final_question: str, option_answers: List[str], std: Literal['mid', 'final'], desc: str, model: str):
        llm_answer = run_llm_prompt(prompt, model=model)
        f1score = f1_score(llm_answer, option_answers)
        verification = ''

        EssEq_results = llm_judge(final_question, golden_answer=option_answers[0], other_answer=llm_answer, llm_judge_prompt=self.prompts["EssEq_prompt"], num_parallel_predictions=1, model=model)
        if std == 'mid' and EssEq_results["avg_score"] >= 1:
            verification = desc
        elif std == 'final' and EssEq_results["avg_score"] < 1:
            verification = desc

        if not verification:
            verification = 'pass'

        return verification, llm_answer, f1score, EssEq_results

    def process_sample_morehop(self, original_question: str, original_answer: str, original_doc: str, topk: int, gen_qa_num: int, num_hop: int, every_hop_qa_num: int, pattern: Literal['simple', 'full']='simple'):
        """
        【主循环：从 1-hop 种子问题扩展到 2/3/4-hop】
        
        参数:
            original_question: 种子问题（如 "谁获得了爱达荷女性历史奖?"）
            original_answer: 种子答案（如 "Cheryl B. Schrader"）
            original_doc: 种子文档
            topk: 每跳返回的文档数（如 10）
            gen_qa_num: 每个文档生成的 QA 数（如 5）
            num_hop: 目标跳数（如 4，表示扩展到 4-hop）
            every_hop_qa_num: 每跳保留的结果数（如 15，避免指数爆炸）
            pattern: 'simple' 只返回最终结果，'full' 返回所有中间结果
        
        返回:
            {"valid_results": [...], "full_results": [...]}
        
        流程:
            hop=1: 种子问题 → 检索 → 抽取 → 合并 → 验证 → 得到 2-hop 问题
            hop=2: 2-hop 问题 → 检索 → 抽取 → 合并 → 验证 → 得到 3-hop 问题
            hop=3: 3-hop 问题 → 检索 → 抽取 → 合并 → 验证 → 得到 4-hop 问题
        """
        # 存储所有符合要求的n_hops的问题
        valid_results = []
        # 存储所有的中间结果
        full_results = []
        # 存储当前hop的合法结果（初始为种子问题）
        current_results = [
            {
                "hop_1": {
                    "question": original_question,
                    "answer": original_answer,
                    "doc": original_doc,
                    "final_question": original_question,
                    "final_answer": original_answer,
                    "refined_answer": original_answer,
                    "qa_type": "initial_qa",
                }
            },
        ]
        
        for hop in range(1, num_hop):  # hop = 1, 2, 3
            temp_results = []
            # 并行处理每个当前结果（10 个线程）
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(self.process_single_data, current_data, hop, topk, gen_qa_num, pattern) for current_data in current_results]
                for fut in as_completed(futures):
                    result, full_result = fut.result()
                    if result:
                        temp_results.extend(result)
                    if full_result and pattern == 'full':
                        full_results.extend(full_result)
            
            print(f"[Info] Completed hop {hop+1}, generated {len(temp_results)} valid intermediate results.")
            
            if hop+1 == num_hop:
                valid_results.extend(temp_results)  # 最终结果
            else:
                # 抽样，避免指数爆炸（每跳只保留 15 个结果继续扩展）
                random.seed(42)
                temp_results = random.sample(temp_results, min(len(temp_results), every_hop_qa_num))
                current_results = temp_results
        
        return {
            "valid_results": valid_results,
            "full_results": full_results
        }

    def process_single_data(self, current_data: Dict[str, Any], hop: int, topk: int, gen_qa_num: int, pattern: str):
        temp_results = []
        full_results = []
        last_hop_data = current_data.get(f"hop_{hop}")
        full_docs = [current_data[f"hop_{h}"]["doc"] for h in range(1, hop+1) if f"hop_{h}" in current_data]
        full_questions = [current_data[f"hop_{h}"]["final_question"] for h in range(1, hop+1) if f"hop_{h}" in current_data]
        full_answers = [current_data[f"hop_{h}"]["refined_answer"] for h in range(1, hop+1) if f"hop_{h}" in current_data]
        full_type = [current_data[f"hop_{h}"]["qa_type"] for h in range(1, hop+1) if f"hop_{h}" in current_data]
        
        retrieved_docs = self.retrieve_docs(
            query=last_hop_data["refined_answer"], 
            original_docs=full_docs, 
            topk=topk, 
            num_hop=hop
        )
        
        for new_doc in retrieved_docs:
            # 【Step 2: 从新文档中抽取 QA 对】
            # 用 gen_qa_prompt 让 LLM 从文档中提取原子级、可验证的事实
            # 如：从 "Cheryl B. Schrader\n...Her presidency was announced on March 6, 2017..."
            # 提取出 {"question": "When was her presidency announced?", "answer": "2017-03-06"}
            gen_prompt = self.prompts["gen_qa_prompt"].format(
                gen_qa_num=gen_qa_num,  # 最多生成 5 个 QA
                input_doc=new_doc,       # 新检索到的文档
            )

            new_qas = run_llm_prompt(gen_prompt, model=self.model_id, return_json=True)  # 调用 gpt-4o-mini
            filter_qas = []
            # print(last_hop_data["refined_answer"])
            # print(new_qas)

            pre_answers = []
            pre_questions = []
            for nq in new_qas:
                #print(nq, type(nq))
                if not isinstance(nq, dict):
                    continue
                question, answer = nq['question'], nq['answer']
                if len(_tokens(answer)) >= 10:
                    #print("too long")
                    continue
                if normalize_answer(answer) in normalize_answer(question):
                    #print("answer in question")
                    continue
                atokens = _tokens(answer)
                qtokens = _tokens(question)
                intokens = [token for token in atokens if token in qtokens]
                if len(intokens) > 0.5 * len(atokens):
                    #print("answer in question")
                    continue
                skip = False
                for pre_answer in pre_answers:
                    if normalize_answer(answer) == normalize_answer(pre_answer):
                        skip = True
                        break
                for pre_question in pre_questions:
                    if normalize_answer(question) == normalize_answer(pre_question):
                        skip = True
                        break
                if skip:
                    #print("duplicate")
                    continue
                else:
                    pre_answers.append(answer)
                    pre_questions.append(question)
                atokens = _tokens(answer)
                qtokens = _tokens(question)
                if "and" in atokens or "or" in atokens or "&" in answer or "and" in qtokens:
                    continue
                if simple_partial_presence(full_answers[-1], question):
                    continue
                if ("full" in qtokens and "name" in qtokens) or ("original" in qtokens and "name" in qtokens) or ("alternate" in qtokens and "name" in qtokens) or ("alternative" in qtokens and "name" in qtokens) or "Name one" in question:
                    continue
                if "document" in qtokens or "article" in qtokens or "according" in qtokens:
                    continue
                filter_qas.append(nq)

            filter_qas = filter_qas[:gen_qa_num]
            for nq in filter_qas:
                mid_question, mid_answer = nq['question'], nq['answer']

                mid_refine_prompt = self.prompts["refine_prompt"].format(
                        question=mid_question,
                        original_answer=mid_answer
                    )
                # 【Step 3: 精炼答案】
                # 统一答案格式（如日期统一为 "March 6, 2017"），方便后续验证
                # 如：{"question": "When was her presidency announced?", "original_answer": "2017-03-06"}
                # → {"refined_answer": "March 6, 2017"}
                mid_refined_result = run_llm_prompt(mid_refine_prompt, model=self.model_id, return_json=True)
                mid_answer = mid_refined_result["refined_answer"]

                skip = False
                for full_answer in full_answers:
                    if normalize_answer(mid_answer) == normalize_answer(full_answer):
                        skip = True
                        break
                if skip:
                    continue

                Data = []
                for h in range(1, hop+1):
                    info = current_data[f"hop_{h}"]
                    Data.append(
                        f"Hop_{h}:\n"
                        f"Question: {info['final_question']}\n"
                        f"Answer: {info['refined_answer']}\n"
                        f"Document: {info['doc']}"
                    )
                if "comparison" not in full_type:
                    # 【Step 4: 合并成多跳问题（inference 类型）】
                    # 用描述性语言替换中间答案，把多个单跳问题"嵌套"成一个多跳问题
                    # 如：Hop_1: "谁获得了爱达荷女性历史奖?" → Cheryl B. Schrader
                    #     New: "她的任期何时宣布?" → March 6, 2017
                    # → Final: "获得爱达荷女性历史奖的当事人的任期何时宣布?" → March 6, 2017
                    merge_prompt = self.prompts["merge_qa_prompt_morehop"].format(
                        max_num=3,
                        Data="\n".join(Data),
                        New_question=mid_question,
                        New_answer=mid_answer,
                        New_document=new_doc,
                    )
                else:
                    # 【Step 4: 合并成多跳问题（comparison 类型）】
                    # 比较两个实体的同一维度（如日期、大小）
                    # 如：Hop_1: "John Beach 何时出生?" → January 1, 1812
                    #     New: "Seth Gordon Persons 何时出生?" → February 5, 1902
                    # → Final: "谁先出生，John Beach 还是 Seth Gordon Persons?" → John Beach
                    merge_prompt = self.prompts["merge_qa_prompt_morehop_comparison"].format(
                        max_num=3,
                        Data="\n".join(Data),
                        New_question=mid_question,
                        New_answer=mid_answer,
                        New_document=new_doc,
                    )

                merged_qas = run_llm_prompt(merge_prompt, model=self.model_id, return_json=True)  # 调用 LLM 合并 
                    
                if isinstance(merged_qas, dict) and merged_qas:
                    merged_qas = [merged_qas]

                for merged in merged_qas:
                    qa_type, final_question, final_answer = merged["type"], merged["final_question"], merged["final_answer"]
                    
                    if qa_type == 'inference':
                        if is_numeric(mid_answer) or (len(_tokens(mid_answer)) == 1 and _years(mid_answer)):
                            continue
                        if normalize_answer(final_answer) != normalize_answer(mid_answer):
                            continue
                        skip = False
                        for pre_question in full_questions:
                            if len(normalize_answer(final_question)) < len(normalize_answer(pre_question)) + 3:
                                skip = True
                                break
                        if skip:
                            continue
                        if normalize_answer(mid_question[:-1]) in normalize_answer(final_question):
                            continue
                    
                    pre_years = []
                    for pre_question in full_questions:
                        pre_years += _years(pre_question)
                    qyear = _years(mid_question)
                    fqyear = _years(final_question)
                    qyear += pre_years
                    if qyear:
                        missing_years = [yr for yr in qyear if yr not in fqyear]
                        if missing_years:
                            # print(f"{mid_question} {qyear}")
                            # print(f"{final_question} {fqyear}")
                            continue

                    skip = False
                    pre_answers = []
                    for h in range(1, hop+1):
                        info = current_data[f"hop_{h}"]
                        if h == 1 or info["qa_type"] == "inference":
                            pre_answers.append(info["final_answer"])
                    for pre_answer in pre_answers:
                        if normalize_answer(pre_answer) in normalize_answer(final_question):
                            skip = True
                            break
                    if skip:
                        #print("leak answer")
                        continue
                    
                    # 精炼答案
                    refine_prompt = self.prompts["refine_prompt"].format(
                        question=final_question,
                        original_answer=final_answer
                    )
                    refined_result = run_llm_prompt(refine_prompt, model=self.model_id, return_json=True)
                    refined_answer = refined_result["refined_answer"]
                    if len(_tokens(refined_answer)) >= 10:
                        continue
                    
                    # 生成更多可选问题
                    opt_prompt = self.prompts["more_optional_answer_prompt"].format(
                        refined_answer=refined_answer
                    )
                    option_answers = run_llm_prompt(opt_prompt, model=self.model_id, return_json=True)

                    # 【Step 5: 四重验证】
                    # 1. 语义检查：确认推理链是否合理（不是错误关联）
                    # 2. 推理检查：让 LLM 独立求解，对比答案（确认问题可解）
                    # 3. 单文档检查：确认单文档无法回答（确认真正需要多跳）
                    # 4. 多文档检查：确认所有文档一起可以回答（确认证据充分）
                    
                    # 构建结果结构
                    new_hop_data = current_data.copy()
                    new_hop_data[f"hop_{hop+1}"] = {
                        "question": mid_question,
                        "answer": mid_answer,
                        "doc": new_doc,
                        "final_question": final_question,
                        "final_answer": final_answer,
                        "refined_answer": refined_answer,
                        "optional_answers": option_answers,
                        "qa_type": qa_type,
                        "verify_result": []
                    }

                    # 验证
                    verification_passed = True
                    verification_steps = []

                    # 1. 语义检查
                    if verification_passed:
                        ## inference
                        if qa_type == 'inference':
                            check_prompt = self.prompts["inference_check_prompt"].format(
                                Question1=last_hop_data["final_question"],
                                Answer1=last_hop_data["refined_answer"],
                                Document1=last_hop_data["doc"],
                                Question2=mid_question,
                                Answer2=mid_answer,
                                Document2=new_doc,
                                Final_question=final_question,
                                Final_answer=final_answer,
                                qa_type=qa_type,
                            )
                        ## comparison
                        else: 
                            check_prompt = self.prompts["comparison_check_prompt"].format(
                                Question1=last_hop_data["final_question"],
                                Answer1=last_hop_data["refined_answer"],
                                Document1=last_hop_data["doc"],
                                Question2=mid_question,
                                Answer2=mid_answer,
                                Document2=new_doc,
                                Final_question=final_question,
                                Final_answer=final_answer,
                                qa_type=qa_type,
                            )

                        check_result = run_llm_prompt(check_prompt, model=self.model_id, return_json=True)
                        verification_steps.append({
                            "step": "multihop_check",
                            "result": check_result,
                            "valid": check_result["valid"].lower()
                        })
                        if check_result["valid"].lower() != 'true':
                            verification_passed = False

                    # 2. reasoning检查
                    if verification_passed:
                        if qa_type == 'inference':
                            reasoning_prompt = self.prompts["reasoning_prompt"].format(
                                problem=final_question
                            )  
                        elif qa_type == 'comparison':
                            reasoning_prompt = self.prompts["comparison_reasoning_prompt"].format(
                                problem=final_question
                            )
                        
                        verification, llm_answer, f1score, EssEq_results = self.compare_verify(
                            prompt=reasoning_prompt,
                            final_question=final_question,
                            option_answers=option_answers,
                            std='mid',
                            desc='reasoning',
                            model=self.model_id
                        )
                        verification_steps.append({
                            "step": "reasoning_check",
                            "verification": verification,
                            "llm_answer": llm_answer,
                            "f1score": f1score,
                            "EssEq_results": EssEq_results,
                            "valid": verification
                        })
                        if verification != 'pass':
                            verification_passed = False

                    # 3. single_hop检查
                    if verification_passed:
                        current_full_docs = full_docs.copy()
                        current_full_docs.append(new_doc)
                        for r in range(len(current_full_docs)-1, len(current_full_docs)):
                            for combo in combinations(current_full_docs, r):
                                if len(combo) == 1:
                                    combo_type = "single_doc"
                                    combo_docs = combo[0]
                                else:
                                    combo_type = f"{len(combo)}_docs_combination"
                                    combo_docs = "\n\n".join(combo)
                                
                                singlehop_prompt = self.prompts["singlehop_prompt"].format(
                                    Document=combo_docs,
                                    Question=final_question,
                                )
                                
                                verification, llm_answer, f1score, EssEq_results = self.compare_verify(
                                    prompt=singlehop_prompt,
                                    final_question=final_question,
                                    option_answers=option_answers,
                                    std='mid',
                                    desc=f'only_{combo_type}',
                                    model=self.model_id
                                )
                                
                                verification_steps.append({
                                    "step": combo_type,
                                    "doc_count": len(combo),
                                    "doc_indices": [current_full_docs.index(doc) for doc in combo],
                                    "verification": verification,
                                    "llm_answer": llm_answer,
                                    "f1score": f1score,
                                    "EssEq_results": EssEq_results,
                                    "valid": verification
                                })
                                
                                if verification != 'pass':
                                    verification_passed = False
                                    break

                    # 4. multihop验证
                    if verification_passed:
                        Data = []
                        for h in range(1, hop+1):
                            info = current_data[f"hop_{h}"]
                            Data.append(
                                f"Question{h}: {info['question']}\n"
                                f"Answer{h}: {info['refined_answer']}\n"
                                f"Supporting Document{h}: {info['doc']}"
                            )
                        if qa_type == 'inference':
                            Data.append(
                                f"Question{hop+1}: {mid_question}\n"
                                f"Supporting Document{hop+1}: {new_doc}"
                            )
                            multihop_prompt = self.prompts["multihop_inference_prompt_morehop"].format(
                                Data="\n".join(Data),
                                FinalQuestion=final_question,
                            )
                        else:
                            Data.append(
                                f"Question{hop+1}: {mid_question}\n"
                                f"Answer{hop+1}: {mid_answer}\n"
                                f"Supporting Document{hop+1}: {new_doc}"
                            )
                            multihop_prompt = self.prompts["multihop_comparison_prompt_morehop"].format(
                                Data="\n".join(Data),
                                FinalQuestion=final_question,
                            )
                        
                        verification, llm_answer, f1score, EssEq_results = self.compare_verify(
                            prompt=multihop_prompt,
                            final_question=final_question,
                            option_answers=option_answers,
                            std='final',
                            desc='cannot_answer',
                            model=self.model_id
                        )
                        verification_steps.append({
                            "step": "full_doc_check",
                            "verification": verification,
                            "llm_answer": llm_answer,
                            "f1score": f1score,
                            "EssEq_results": EssEq_results,
                            "valid": verification
                        })
                        if verification != 'pass':
                            verification_passed = False

                    new_hop_data[f"hop_{hop+1}"]["verify_result"] = verification_steps

                    if verification_passed:
                        temp_results.append(new_hop_data)
                    
                    if pattern == 'full':
                        full_results.append(new_hop_data)
                        
        return temp_results, full_results