"""
FastAPI HTTP layer for the market environment.

Exposes the standard OpenEnv endpoints on top of MarketEnvironment.

Endpoints:
    GET  /health        — liveness probe
    GET  /tasks         — list available task configurations
    POST /reset         — start a new episode, return initial observation
    POST /step          — advance an episode by one turn
    GET  /state         — query the current state of an episode

Run locally:
    uvicorn market_env.server:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import pathlib
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from market_env.environment import (
    EpisodeAlreadyDone,
    EpisodeNotFound,
    MarketEnvironment,
)
from market_env.models import MarketAction, MarketObservation


app = FastAPI(
    title="Multi-Agent Market RL Environment",
    description=(
        "OpenEnv-compliant trading environment for theory-of-mind LLM training. "
        "Agents trade on a continuous double-auction order book with asymmetric "
        "private information; profit signal is the primary reward."
    ),
    version="0.1.0",
)

# Single env instance shared across all requests; manages multiple sessions internally.
env = MarketEnvironment()


# ---------------------------------------------------------------------------
# Request/response wrappers
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: Optional[str] = None
    seed: int = 42
    difficulty: str = "medium"
    bot_config: str = "default"
    trainable_agent_id: str = "agent_1"
    episode_length: int = 50


class StepRequest(BaseModel):
    episode_id: str
    action: MarketAction


class StepResponse(BaseModel):
    observation: MarketObservation
    reward: float
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> HTMLResponse:
    """Serve the blog / writeup as the landing page."""
    blog_path = pathlib.Path(__file__).parent.parent / "blog.md"
    if blog_path.exists():
        md = blog_path.read_text(encoding="utf-8")
    else:
        md = "# Multi-Agent Market RL\n\nSee /docs for the API."

    # Minimal markdown → HTML conversion (headings, bold, code, tables, hr, links)
    import re

    def _md_table_to_html(block: str) -> str:
        lines = [l.rstrip() for l in block.strip().splitlines()]
        rows_html: list[str] = []
        is_header = True
        for line in lines:
            # Skip GFM separator rows like |---|:---:|---
            stripped = line.replace("|", "").replace("-", "").replace(":", "").replace(" ", "")
            if not stripped:
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            tag = "th" if is_header else "td"
            row = "".join(f"<{tag}>{c}</{tag}>" for c in cells)
            rows_html.append(f"<tr>{row}</tr>")
            is_header = False
        return "<table>" + "".join(rows_html) + "</table>"

    html_body = md
    # Fenced code blocks
    html_body = re.sub(r"```(?:\w+)?\n(.*?)```", lambda m: f"<pre><code>{m.group(1)}</code></pre>", html_body, flags=re.DOTALL)
    # GFM tables — convert before paragraph wrapping so newlines are still intact
    html_body = re.sub(
        r"(?m)^(\|.+\|\n)+",
        lambda m: _md_table_to_html(m.group(0)) + "\n\n",
        html_body,
    )
    # Inline code
    html_body = re.sub(r"`([^`]+)`", r"<code>\1</code>", html_body)
    # Images
    html_body = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r'<img src="\2" alt="\1" style="max-width:100%;border-radius:8px;">', html_body)
    # Links
    html_body = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', html_body)
    # Bold
    html_body = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", html_body)
    # Italic
    html_body = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", html_body)
    # H1-H3
    html_body = re.sub(r"^### (.+)$", r"<h3>\1</h3>", html_body, flags=re.MULTILINE)
    html_body = re.sub(r"^## (.+)$", r"<h2>\1</h2>", html_body, flags=re.MULTILINE)
    html_body = re.sub(r"^# (.+)$", r"<h1>\1</h1>", html_body, flags=re.MULTILINE)
    # HR
    html_body = re.sub(r"^---$", "<hr>", html_body, flags=re.MULTILINE)
    # Paragraphs (blank-line separated)
    html_body = re.sub(r"\n{2,}", "</p><p>", html_body)
    html_body = f"<p>{html_body}</p>"

    return HTMLResponse(content=f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Multi-Agent Market RL — Theory of Mind Environment</title>
<style>
  body {{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
         max-width:860px;margin:0 auto;padding:2rem 1.5rem;
         background:#0d1117;color:#c9d1d9;line-height:1.7;}}
  h1,h2,h3{{color:#f0f6fc;margin-top:2rem;}}
  h1{{font-size:2rem;border-bottom:1px solid #21262d;padding-bottom:.5rem;}}
  h2{{font-size:1.4rem;border-bottom:1px solid #21262d;padding-bottom:.3rem;}}
  a{{color:#58a6ff;text-decoration:none;}} a:hover{{text-decoration:underline;}}
  code{{background:#161b22;padding:.15em .4em;border-radius:4px;
        font-family:'SFMono-Regular',Consolas,monospace;font-size:.9em;color:#e6edf3;}}
  pre{{background:#161b22;padding:1rem;border-radius:8px;overflow-x:auto;
       border:1px solid #21262d;}}
  pre code{{background:none;padding:0;}}
  table{{border-collapse:collapse;width:100%;margin:1rem 0;}}
  td,th{{border:1px solid #21262d;padding:.5rem .75rem;text-align:left;}}
  tr:nth-child(even){{background:#161b22;}}
  strong{{color:#f0f6fc;}}
  hr{{border:none;border-top:1px solid #21262d;margin:2rem 0;}}
  img{{border:1px solid #21262d;}}
  .api-bar{{background:#161b22;border:1px solid #21262d;border-radius:8px;
             padding:.75rem 1rem;margin-bottom:2rem;font-size:.9rem;}}
  .api-bar a{{margin-right:1rem;}}
</style>
</head>
<body>
<div class="api-bar">
  <strong>Live API:</strong>
  <a href="/docs">📖 Docs (Swagger)</a>
  <a href="/health">❤️ Health</a>
  <a href="/tasks">📋 Tasks</a>
  <a href="https://github.com/PrathameshWable/market-rl-env">GitHub</a>
  <a href="https://colab.research.google.com/drive/1dVUBw60a5JrGvVYdcL3wdZVQ1QGfXnre?usp=sharing">Colab Notebook</a>
</div>
{html_body}
</body>
</html>""")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/tasks")
def list_tasks() -> list[dict[str, Any]]:
    return env.list_tasks()


@app.post("/reset", response_model=MarketObservation)
def reset(req: Optional[ResetRequest] = None) -> MarketObservation:
    if req is None:
        req = ResetRequest()
    try:
        return env.reset(
            task_id=req.task_id,
            seed=req.seed,
            difficulty=req.difficulty,
            bot_config=req.bot_config,
            trainable_agent_id=req.trainable_agent_id,
            episode_length=req.episode_length,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest) -> StepResponse:
    try:
        obs, reward, done, info = env.step(req.episode_id, req.action)
    except EpisodeNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except EpisodeAlreadyDone as e:
        raise HTTPException(status_code=400, detail=str(e))
    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/state")
def get_state(episode_id: str = Query(...)) -> dict[str, Any]:
    try:
        return env.state(episode_id)
    except EpisodeNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
