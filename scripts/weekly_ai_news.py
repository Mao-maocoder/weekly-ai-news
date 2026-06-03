import os
import json
import datetime as dt
from pathlib import Path

import requests
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports"
OUT_DIR.mkdir(exist_ok=True)

today = dt.date.today()
since = today - dt.timedelta(days=7)
report_path = OUT_DIR / f"{today.isoformat()}.md"

github_token = os.environ.get("GITHUB_TOKEN", "")
headers = {
    "Accept": "application/vnd.github+json",
}
if github_token:
    headers["Authorization"] = f"Bearer {github_token}"

topics = [
    "topic:llm",
    "topic:generative-ai",
    "topic:artificial-intelligence",
    "topic:machine-learning",
    "topic:ai-agents",
    "topic:rag",
]

repos = {}

for topic in topics:
    params = {
        "q": f"{topic} stars:>=50 pushed:>={since.isoformat()} archived:false",
        "sort": "stars",
        "order": "desc",
        "per_page": 10,
    }
    resp = requests.get(
        "https://api.github.com/search/repositories",
        headers=headers,
        params=params,
        timeout=30,
    )
    resp.raise_for_status()

    for item in resp.json().get("items", []):
        repos[item["full_name"]] = {
            "name": item["full_name"],
            "url": item["html_url"],
            "description": item.get("description") or "",
            "stars": item["stargazers_count"],
            "forks": item["forks_count"],
            "language": item.get("language") or "",
            "updated_at": item["updated_at"],
            "topics": item.get("topics", []),
        }

top_repos = sorted(repos.values(), key=lambda x: x["stars"], reverse=True)[:30]

client = OpenAI(
    api_key=os.environ["MODEL_API_KEY"],
    base_url=os.environ["MODEL_BASE_URL"],
)

prompt = f"""
请基于下面这些 GitHub 仓库，生成一份中文 AI 热点周报。

时间范围：{since.isoformat()} 到 {today.isoformat()}

要求：
1. 先给出 5 条本周趋势判断。
2. 再列出 10 个最值得关注的项目。
3. 每个项目包含：项目名、链接、star 数、主要看点、适合谁关注。
4. 最后给出一个“下周值得继续观察”的列表。
5. 不要编造 GitHub 数据之外的事实。

数据：
{json.dumps(top_repos, ensure_ascii=False, indent=2)}
"""

response = client.chat.completions.create(
    model=os.environ["MODEL_NAME"],
    messages=[
        {"role": "user", "content": prompt}
    ],
)
text = response.choices[0].message.content

report_path.write_text(text, encoding="utf-8")
print(f"Wrote {report_path}")