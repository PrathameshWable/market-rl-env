# Milestones — Multi-Agent Market RL (18-Hour Sprint Edition)
# Revised: April 25, 2026 — 18 hours to submission

## What's done

| Milestone | Status | Notes |
|-----------|--------|-------|
| M1 — Order book engine | ✅ DONE | 59 tests, 100% coverage |
| M2 — Scenario + scripted bots | ✅ DONE | 5 bots, full episode sim |
| M3 — Environment + server + client | ✅ DONE | 213 tests, 99% coverage |
| M4 — Docker + HF Space deployment | ✅ DONE | https://huggingface.co/spaces/Prathamesh0292/market-rl-env |
| M5 — SFT + GRPO training | 🔄 RUNNING | GRPO on Colab, step ~60/300, ~1.5h left |

---

## 18-Hour Execution Plan

**Deadline: ~April 26, 2026 evening**

### NOW — While training is still running (Hours 0–1.5)

- [ ] Push all local commits to HF Space: `git push space main --force`
- [ ] Verify Space rebuilds and `/health` still returns 200
- [ ] Start writing `blog.md` (do in parallel — it takes 2–3 hours)

---

### M5 completion — after training finishes (Hours 1.5–3)

- [ ] Confirm checkpoint saved to Drive at `/MyDrive/market-rl-stage1/grpo-checkpoint`
- [ ] Run smoke test cell (5 episodes, report mean pnl_normalized)
- [ ] Download reward curve PNG → commit to `assets/results/reward_curve_stage1.png`
- [ ] Make Colab notebook public: **File → Share → Anyone with link → Viewer**
- [ ] Copy public Colab URL → add to README

**Done when:** Colab link is public. Reward curve committed. Smoke test shows model is trading.

---

### M6 — Evaluation (simplified, Hours 3–5)

No statistical significance. No ToM probes. Just enough for real numbers.

- [ ] Write `training/evaluate.py` — runs model on N scenarios, returns metrics dict
- [ ] Run eval on **10 held-out scenarios** (seeds 100–109, medium difficulty)
- [ ] Record 3 numbers: mean normalized P&L, participation rate, parse success rate
- [ ] Run same 10 scenarios with RandomBot as baseline
- [ ] Commit `assets/results/eval_results.json`

**Done when:** You have 3 numbers for trained model and 3 for random baseline.

**Cut from original M6:** ToM probes, statistical significance, 50-scenario eval, price efficiency. All labelled "future work" in blog.

---

### M7 — SKIPPED

No time. Stage 2 self-play and ablations are cut. Best Stage 1 checkpoint IS the submission.

---

### M8 — Blog + README + Submit (Hours 5–14)

#### blog.md (Hours 5–8)

Write at repo root as `blog.md`. Also publish to HF Blog.

Sections (stick to this, ~600 words total):
1. **Hook** — "We gave a language model $10,000 and four opponents, each knowing something different about the stock's true value..."
2. **Environment** — 1 paragraph + show `MarketObservation` fields snippet
3. **Training** — SFT warm-start (why) + GRPO (what) + reward formula
4. **Results** — 1 table (random vs. trained) + embed reward curve PNG
5. **What's next** — ToM probes, Stage 2, mention as future work
6. **Links** — HF Space, GitHub, Colab

Tasks:
- [ ] Write `blog.md`
- [ ] Publish to HF Blog (paste markdown, upload reward curve image)
- [ ] Copy HF Blog URL

#### README.md (Hours 8–9)

Fill ALL 4 placeholder URLs:
- [ ] Colab notebook link (after M5)
- [ ] Code repo link (HF Space repo or GitHub)
- [ ] Blog URL (after publishing)
- [ ] Embed reward curve image
- [ ] Add 2-row eval table (random vs. trained)
- [ ] Check all links open correctly

#### GitHub push (Hours 9–9.5)

- [ ] Create repo at github.com/PrathameshWable/market-rl-env (web UI)
- [ ] `git push origin main`
- [ ] Use GitHub URL as the "code repository link"

#### Final checks (Hours 9.5–10)

- [ ] `curl https://prathamesh0292-market-rl-env.hf.space/health` → 200
- [ ] All 4 README links open in browser
- [ ] `pytest tests/ -q` passes clean
- [ ] Space shows "Running"

---

## Hour-by-hour breakdown

| Hours from now | Task | Done when |
|----------------|------|-----------|
| 0–1.5 | Push to Space + start blog.md | blog draft started |
| 1.5–3 | Training finishes → smoke test → reward curve → notebook public | Colab link copied |
| 3–5 | M6 eval (10 scenarios, 3 metrics) | eval_results.json committed |
| 5–8 | Write blog.md + publish to HF Blog | Blog URL in hand |
| 8–9 | README: all 4 URLs + reward curve + eval table | No TODOs in README |
| 9–10 | GitHub push + final checks | All links verified |
| 10–18 | Buffer: sleep, fix issues, submit form | Form submitted |

---

## Submission checklist (put all 4 in the README)

- [ ] HF Space URL: `https://huggingface.co/spaces/Prathamesh0292/market-rl-env` ✅
- [ ] Colab notebook link: `[fill after training finishes]`
- [ ] Code repo link: `[fill after GitHub push]`
- [ ] Blog/video URL: `[fill after blog is published]`

**Minimum bar to submit:**
- Space Running, `/health` → 200 ✅
- Reward curve plot in README
- Any eval numbers (even 5 episodes)
- Colab link (public, opens without login)
- Blog OR video URL
