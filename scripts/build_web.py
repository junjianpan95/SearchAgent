"""把 demo 轨迹数据注入前端 HTML，生成自包含的单文件前端。"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAJ = ROOT / "data" / "demo_trajectories.json"
HTML_TEMPLATE = ROOT / "web" / "template.html"
OUT = ROOT / "web" / "index.html"

data = json.loads(TRAJ.read_text(encoding="utf-8"))
data_json = json.dumps(data, ensure_ascii=False)

html = HTML_TEMPLATE.read_text(encoding="utf-8")
html = html.replace("__DEMO_DATA__", data_json)
OUT.write_text(html, encoding="utf-8")
print(f"[WEB] 生成前端页面 -> {OUT}（含 {len(data)} 条 demo 轨迹）")
