"""
weekly_ai_news.py
每周自动抓取 GitHub AI 热点仓库，并用 LLM 生成中文周报。

设计要点：
1. 多 topic 分别查询，避免 GitHub Search 不支持 topic OR 导致结果为空。
2. 在 reports/_snapshots/ 保存每周快照，使现有 `git add reports/` 可提交快照。
3. 优先按 7 日 Star 增量排序；没有历史快照时回退到总 Star 排序。
"""

import datetime as dt
import json
import logging
import os
from pathlib import Path
from typing import Any

import requests
from openai import OpenAI

# -- 配置 ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports"
SNAPSHOT_DIR = OUT_DIR / "_snapshots"
OUT_DIR.mkdir(exist_ok=True)
SNAPSHOT_DIR.mkdir(exist_ok=True)

TOP_N = 30
MIN_STARS = 50
PER_TOPIC_LIMIT = 30
TREND_COUNT = 5
HIGHLIGHT_COUNT = 10

TOPICS = [
    "topic:llm",
    "topic:generative-ai",
    "topic:artificial-intelligence",
    "topic:machine-learning",
    "topic:ai-agents",
    "topic:rag",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# -- 日期 ---------------------------------------------------------------------

today = dt.date.today()
since = today - dt.timedelta(days=7)
report_path = OUT_DIR / f"{today.isoformat()}.md"
snapshot_path = SNAPSHOT_DIR / f"{today.isoformat()}.json"

# -- GitHub 请求头 -------------------------------------------------------------

github_token = os.environ.get("GITHUB_TOKEN", "")
gh_headers = {
    "Accept": "application/vnd.github+json",
    "User-Agent": "weekly-ai-news",
}
if github_token:
    gh_headers["Authorization"] = f"Bearer {github_token}"
else:
    log.warning("未设置 GITHUB_TOKEN，使用未认证请求（每小时限 60 次）")


def fetch_repos(topics: list[str], since_date: dt.date, min_stars: int) -> dict[str, dict[str, Any]]:
    """按 topic 分别查询 GitHub 仓库，并用 full_name 去重。"""
    repos: dict[str, dict[str, Any]] = {}

    for topic in topics:
        params = {
            "q": f"{topic} stars:>={min_stars} pushed:>={since_date.isoformat()} archived:false",
            "sort": "stars",
            "order": "desc",
            "per_page": PER_TOPIC_LIMIT,
        }

        log.info("正在请求 GitHub Search API: %s", topic)
        try:
            resp = requests.get(
                "https://api.github.com/search/repositories",
                headers=gh_headers,
                params=params,
                timeout=30,
            )
            log.info("GitHub API rate-limit 剩余：%s", resp.headers.get("X-RateLimit-Remaining", "?"))
            resp.raise_for_status()
        except requests.HTTPError:
            log.error("GitHub API 请求失败：status=%s, body=%s", resp.status_code, resp.text[:500])
            raise
        except requests.RequestException as exc:
            log.error("GitHub API 网络错误：%s", exc)
            raise

        for item in resp.json().get("items", []):
            repos[item["full_name"]] = {
                "name": item["full_name"],
                "url": item["html_url"],
                "description": item.get("description") or "",
                "stars": item["stargazers_count"],
                "forks": item["forks_count"],
                "language": item.get("language") or "",
                "created_at": item["created_at"],
                "updated_at": item["updated_at"],
                "pushed_at": item["pushed_at"],
                "topics": item.get("topics", []),
            }

    log.info("共获取 %s 个唯一仓库", len(repos))
    return repos


def load_previous_snapshot(snapshot_dir: Path, before_date: dt.date) -> dict[str, Any] | None:
    """读取最近一次早于今天的快照，用于计算 Star 增量。"""
    candidates = sorted(snapshot_dir.glob("*.json"), reverse=True)
    for path in candidates:
        try:
            snapshot_date = dt.date.fromisoformat(path.stem)
        except ValueError:
            continue
        if snapshot_date >= before_date:
            continue
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log.warning("忽略无法解析的快照：%s", path)
    return None


def add_star_deltas(
    repos: dict[str, dict[str, Any]],
    previous_snapshot: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """给仓库补充 stars_delta_7d；无历史数据时为 None。"""
    previous_repos = {}
    if previous_snapshot:
        previous_repos = previous_snapshot.get("repos", {})

    enriched = []
    for name, repo in repos.items():
        old_repo = previous_repos.get(name)
        old_stars = old_repo.get("stars") if isinstance(old_repo, dict) else None
        stars_delta = repo["stars"] - old_stars if isinstance(old_stars, int) else None
        enriched.append(
            {
                **repo,
                "stars_delta_7d": stars_delta,
                "is_new_in_candidates": old_repo is None,
            }
        )
    return enriched


def rank_repos(repos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """优先按本周 Star 增量排序；首周没有快照时按总 Star 回退。"""
    has_delta = any(repo["stars_delta_7d"] is not None for repo in repos)
    if has_delta:
        return sorted(
            repos,
            key=lambda repo: (
                repo["stars_delta_7d"] if repo["stars_delta_7d"] is not None else -1,
                repo["stars"],
            ),
            reverse=True,
        )
    return sorted(repos, key=lambda repo: repo["stars"], reverse=True)


def save_snapshot(path: Path, repos: dict[str, dict[str, Any]]) -> None:
    """保存当前候选仓库快照，供下次运行计算增量。"""
    snapshot = {
        "date": today.isoformat(),
        "since": since.isoformat(),
        "topics": TOPICS,
        "repos": repos,
    }
    path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("快照已写入：%s", path)


repos_by_name = fetch_repos(TOPICS, since, MIN_STARS)
previous_snapshot = load_previous_snapshot(SNAPSHOT_DIR, today)
if previous_snapshot:
    log.info("已读取历史快照：%s", previous_snapshot.get("date", "unknown"))
else:
    log.info("未找到历史快照，本次将按总 Star 排序")

ranked_repos = rank_repos(add_star_deltas(repos_by_name, previous_snapshot))
top_repos = ranked_repos[:TOP_N]
save_snapshot(snapshot_path, repos_by_name)

# -- 调用模型生成周报 ----------------------------------------------------------

required_envs = ["MODEL_API_KEY", "MODEL_BASE_URL", "MODEL_NAME"]
missing_envs = [name for name in required_envs if not os.environ.get(name)]
if missing_envs:
    raise RuntimeError(f"缺少必填环境变量：{', '.join(missing_envs)}")

client = OpenAI(
    api_key=os.environ["MODEL_API_KEY"],
    base_url=os.environ["MODEL_BASE_URL"],
)

ranking_note = (
    "本周候选项目优先按 stars_delta_7d（相对上一份快照的 Star 增量）排序。"
    if previous_snapshot
    else "本周是首次生成或没有历史快照，因此候选项目按总 Star 排序；下周开始会优先使用 Star 增量。"
)

prompt = f"""请基于下面这些 GitHub 仓库，生成一份中文 AI 热点周报。

时间范围：{since.isoformat()} 到 {today.isoformat()}
排序说明：{ranking_note}

要求：
1. 标题使用：# AI 热点周报 ({since.isoformat()} 至 {today.isoformat()})
2. 先给出 {TREND_COUNT} 条本周趋势判断，每条都要引用至少 1 个仓库作为依据。
3. 再列出 {HIGHLIGHT_COUNT} 个最值得关注的项目。
4. 每个项目包含：项目名、链接、Star 数、本周 Star 增量（如果为 null 就写“暂无历史快照”）、主要看点、适合谁关注。
5. 最后给出“下周值得继续观察”列表（3 到 5 条）。
6. 严格基于所给数据，不要编造 GitHub 数据之外的事实。
7. 如果 stars_delta_7d 为 null，不要声称该项目本周增长很快，只能说它总 Star 高且最近仍活跃。

数据：
{json.dumps(top_repos, ensure_ascii=False, indent=2)}
"""

log.info("正在调用模型生成周报 …")
response = client.chat.completions.create(
    model=os.environ["MODEL_NAME"],
    max_tokens=4096,
    messages=[{"role": "user", "content": prompt}],
)
text = response.choices[0].message.content or ""

if not text.strip():
    raise RuntimeError("模型返回内容为空")

report_path.write_text(text, encoding="utf-8")
log.info("周报已写入：%s", report_path)
