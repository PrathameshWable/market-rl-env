# Recording the Meta-Trader video

A 90-second walkthrough hits everything a judge needs. Below is a prep
checklist and a tight script.

## 0. Setup (~2 min)

```powershell
cd D:\Meta_ROUND2
python -m pip install -r requirements-dashboard.txt
streamlit run dashboard/app.py
```

Open `http://localhost:8501` in **Chrome / Edge in fullscreen** (F11). The
contrast on the dark theme is best at 1920×1080. Use OBS or the built-in
Windows Game Bar (`Win + Alt + R`) for screen capture.

In the sidebar, pick the run you want to feature (e.g. `stage1_2026-04-25`
for M6, `aux_direction` once M7B finishes).

## 1. Recording script (≈ 90 seconds)

| Time  | Section in app          | What to say                                                                                                                                                                               |
| ----- | ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0–8s  | **Overview** (top cards)| "Stage 1 of a Theory-of-Mind market trading agent. Qwen-3B fine-tuned with SFT then GRPO on a multi-bot order-book environment."                                                          |
| 8–20s | **Overview** (KPI row)  | "+5.5% mean P&L vs random's −3.5%. 100% parse rate. The model moves the order book toward truth — price-efficiency error drops from 2.83 to 2.18 dollars per share over the episode."     |
| 20–30s| **Overview** (story cards) | "Eighty percent of the agent's trades go in the direction implied by its private signals. But probe 3 — direction inference — is still 50%, the chance baseline. M7B fixes that."     |
| 30–60s| **Live Market Replay**  | Pick `seed = 100`, surrogate = **Informed**. Click `Run scenario` then `Auto-play`. "Same scenario, same bots, swap only the trainable agent. The order book converges to the green dashed line — that's the true value." Re-run with surrogate = **Random** to show how the dumb baseline never gets there. |
| 60–75s| **Theory of Mind**      | Scroll to the price-efficiency curve. "Trained agent in green tracks closer to informed than to random." Then to probe 3 confusion matrix. "The 'say above' bias visualised — that's exactly the failure M7B's auxiliary direction reward is designed to break." |
| 75–90s| **Training Curves** OR **Methodology** | "Two hundred GRPO steps; reward stabilised around plus-zero-five." Drop to Methodology to flash the JSON action format. "All artifacts ship as JSON in `training/runs/` — every number on screen is reproducible." |

## 2. Live-replay tips

- The **Auto-play** button advances frames at ~5fps. If your screen
  recorder is 30fps, the playback looks smooth.
- The **green dashed line is the true value**. The story is "watch the mid
  price converge to that line under the trained surrogate, but not under
  random." Pick a seed where the spread between true value and the
  initial mid is large — `seed = 104` (true value 40.41, big gap) is
  great.
- The order-book table on the right updates per-turn. Pause on it during
  narration to point at bid/ask quantities being deliberately stacked
  on one side after the agent's signal-aligned trade.
- Drag the **Turn slider** if you want to scrub backwards for a callback.

## 3. Recommended seeds for visual impact

| Seed | True value | What you'll see                                                       |
|------|-----------:|-----------------------------------------------------------------------|
| 100  | 48.93      | Subtle below-50; trained vs random separation is moderate            |
| 104  | 40.41      | Big bear scenario — informed wins big, random tanks                  |
| 105  | 55.55      | Bull scenario; trained finishes ~+8% on this seed                     |
| 109  | 47.95      | Mild bear, good for showing signal-alignment trades                   |

## 4. Pre-flight before pressing record

- [ ] `streamlit run dashboard/app.py` is running and the page loads at 1920×1080
- [ ] Sidebar run selector is on the run you want to feature
- [ ] On the **Live Replay** tab, click `Run scenario` once so the in-page state
      is initialised (otherwise the first auto-play starts cold)
- [ ] Mute notifications; close Slack/email
- [ ] OBS scene is set to capture `Chrome` only (not the whole desktop)

## 5. Post-record

Trim head/tail in any editor (Clipchamp, DaVinci Resolve), export at
1080p / 30fps / H.264, target file size **<50 MB**. That's small enough
to attach to the submission directly.
