# Invest- Day-Trading Bot — Veteran Trader Review & Improvement Plan (rev 2, commit 8ed1570)

## Context

The owner wants the IBKR auto day-trader in `gsajang-cpu/Invest-` (`ibkr-day-trader/`) to make 20–22%. I reviewed the latest commit `8ed1570` ("Oct 30 trading release"): strategy, risk, engine, paper runner, configs, and the new Oct 30 plan, status and release docs.

**What changed since my first review**
- The paper reconciliation fault is resolved and **paper is trading again**: 14 closed trade groups and 26 fills. The Oct 29 end-of-day result was 5 trades, net ≈ −$0.51 (≈ 0R). 2 of 20 qualifying sessions are done.
- Paper capacity was raised to **20 positions and 20 entries a day** (`config/paper-policy.json` schema 3, `paper_pooled_capital`).
- The 11-ticket Oct 30 program is deployed: data diagnostics, session-history repair, readiness-v2, reconciliation checkpoints, cost research CLI, admission reports, two research-only strategies (VWAP pullback and breakout retest), day-block bootstrap performance contract, broker drills, release bundles.
- **The production strategy is unchanged.** `strategy.py` only gained a diagnostics wrapper. `risk.py` and the bracket exits (fixed 2R, no trail) are untouched. The live config is still the $50 pilot.

**Bottom line:** the Oct 30 program is excellent *plumbing and research governance*. It deliberately leaves the trading logic alone ("no production threshold change"). That trading logic is where the money is lost or made. This revision keeps only what the Oct 30 plan does **not** already cover, so nothing is duplicated.

**Scope:** review and plan only — no Invest- code has been changed. Each phase needs owner approval before implementation.

---

## 1. The target: daily or annual? Both need resetting

- The project docs now say **"20–22% annual (aspirational)"**. The request says **20–22% daily**.
- **Daily is impossible.** Compounded, it turns $2,000 into roughly 10^23 dollars in a year. The bot's own caps limit a perfect day to about 6% ($10 risk × 2R × 6 winners); even 20 winning slots is capped by the $40 daily loss rule on the way down and by cash on the way up.
- **Annual 20–22% is ambitious but legitimate**, about +$400 a year on $2,000. Fees alone work against it: at the $1 Fixed minimum, every round trip costs about $2, which is 0.1% of the account. 200 trades a year means 20% of the account goes to commissions, so the strategy must beat its own costs by a wide margin.
- **KPIs to adopt:** expectancy after costs > +0.15R over at least 200 trades, profit factor ≥ 1.3, max drawdown ≤ 10%, average cost per trade < 0.2R.

## 2. Flaws the Oct 30 plan does not address (veteran trader view)

### Capital, regulation, capacity
| # | Flaw | Evidence | Fix |
|---|---|---|---|
| C1 | **20/20 capacity on $2,000 is fake diversification.** At $500 max notional, cash fits only about 4 positions. 20 slots just let the bot take 20 marginal, correlated long-momentum trades. All of them are "SPY > VWAP" longs, so it's really one bet | `paper-policy.json`, `config.py` `paper_pooled_capital` | Cap at 3–4 concurrent positions and 4–6 entries a day; add a sector/correlation cap (activate P1-06's research constraint) |
| C2 | **PDT (pattern day trader) rule**: with live under $25k on margin, 20 or even 6 entries a day is impossible. The paper results won't carry over to live | `account_policy.py` still `legacy_pdt` with live regime "unsupported"; `service_contracts.live_admission_check` only needs ≥ 1 | Paper should *simulate* the live constraint (day-trade budget), or the paper evidence is invalid for live. Plan live as ≤ 1 A+ trade a day, or as a cash account, or at ≥ $25k |
| C3 | The live config is still the $50 pilot, which can't trade (0 plans) | `config/pilot.json` | Retire it; the live profile should mirror the paper $2,000 profile at 0.25% risk |

### Strategy (`strategy.py`, unchanged in 8ed1570)
| # | Flaw | Veteran fix |
|---|---|---|
| S1 | **Universe isn't "in play."** The scanner is MOST_ACTIVE plus TOP_PERC_GAIN, rotating 20 streams (`discovery.py`). ORB edge lives in gappers with a catalyst | Pre-market (8:00–9:29 ET) gap scan: gap ≥ 2–3%, pre-market volume, true RVOL ≥ 2, ATR% ≥ 2, price $5–$200, news/earnings flag. **Freeze a 5–15 name watchlist at 9:29.** This also fixes the rotation and backfill issues that P0-02 is fighting |
| S2 | **"Relative volume" isn't RVOL**: last 1-min bar ÷ median of today's previous 10 bars (`recent = ordered[-11:-1]`) | Cumulative volume vs. the 10–20 day average at the same time of day |
| S3 | **Entry window 9:45–15:30** is far too long for ORB | `entry_cutoff`, default 11:00 ET (backtest to confirm) |
| S4 | **Arbitrary stop**: `max(0.3% × price, 0.35 × OR width)`, inside the noise | Structural stop: below the breakout bar low or the OR midpoint, with a floor of k × 1-min ATR |
| S5 | **Percent spread cap (0.15% live / 0.25% paper)** rules out good sub-$10 names and doesn't relate to the stop | Spread ≤ 10–15% of the stop distance, plus a minimum dollar volume |
| S6 | **Late, chasing entry**: waits for a 1-min close, then a limit at the current ask with a 120 s timeout; entry slippage isn't modeled | Resting buy-stop-limit at OR high + 1–2 ticks, short expiry; add entry slippage to `RiskEngine.round_trip_cost` |
| S7 | **No trade management.** Exits are a broker bracket with fixed 2R LMT plus STP; `controls.py` "trailing … implementation pending" is still there. The research strategies only change entries | Exit manager: ½ off at 1R and stop to breakeven; trail the rest (1-min/5-min higher lows or VWAP); time stop at 10–15 minutes without reaching +0.5R; flatten at close-out (already there). Needs stop modification and partial-close support in the broker adapters and `broker_contract.protected` |
| S8 | **Ranking by cost efficiency.** `engine.py` `priority` sorts by net R:R, which with a fixed 2R target just favors the lowest cost | Rank by setup quality (RVOL, gap + catalyst, OR tightness vs ATR, relative strength vs SPY/sector); keep R:R as a gate |
| S9 | Only filter is SPY > VWAP; no macro calendar | Skip the first minutes after CPI/FOMC/NFP; VIX regime band; promote `spy_or_sector_rs` only if it backtests better |
| S10 | Long only | Later: opening-range-breakdown shorts (margin, borrow availability) |

### Risk (`risk.py`)
Add: a **consecutive-loss stop** (2 straight losers or 3 total ends the day), a **green-day lock** (after +2R, stop if 50% of the gain is given back), **percent-of-equity sizing** (0.25–0.5%), a per-sector cap, and a PDT budget gate.

### Process
- The 20 paper sessions × 0–5 trades a day is too little for statistics, as P2-09 already admits ("inconclusive"). **Buy 12–24 months of historical 1-min data now** (the in-play universe, including delisted names) and run the existing replay/LEAN harness. Use paper only to validate *execution quality* (modeled vs actual fills).
- The 14 trades so far are around 0R. Start the trade journal review now: MAE/MFE per trade, how many hit 1R and then reversed to the stop. That tells you directly whether S7 (trade management) is the biggest lever.

## 3. Implementation plan (builds on the Oct 30 tickets; one ticket per session, same contract)

**Phase 0 — Reset (week 1)**
- T0-1: Rewrite the goal as the KPIs (README, `memory.md`).
- T0-2: Reduce paper capacity to 4/6 (`paper-policy.json`); retire the $50 live config.
- T0-3: Simulated PDT budget in paper (`risk._global_block`, `service_contracts`).
- T0-4: Trade journal analytics (MAE/MFE, R-multiples, by time of day) on the existing 14+ trades (`opportunity_research.py`, `weekly_review.py`).

**Phase 1 — Historical backtest foundation (weeks 1–3)**
Data loader for purchased 1-min and daily data → existing `replay.py` and `research/lean` → P2-09 performance contract. Baseline the current ORB.

**Phase 2 — Strategy A/B, one change at a time (weeks 2–5)**
S1 watchlist (`discovery.py`, `market_data.py`) → S2 RVOL → S3 cutoff → S4 stop → S5 spread → S8 ranking (`engine.py`) → S6 entry order (`ibkr.py`, `paper_broker.py`) → S7 exit manager (new `exit_manager.py`, wired into `paper_runner.supervise_exits`, `broker_contract.py`) → S9 regime filter. Keep a change only if out-of-sample expectancy improves and survives 2× costs, using the existing frozen holdout.

**Phase 3 — Risk additions (week 4)**
`risk.py` plus `RiskState` in `models.py`.

**Phase 4 — Forward paper (≥ 20 sessions)**
Track execution quality vs the model; stop if results fall outside the backtest's 95% band.

**Phase 5 — Live canary**
$2,000+, 0.25% risk, PDT-limited. Scale only after 100+ live trades meet the KPIs.

**Phase 6 (optional)**
Short side; promote the pullback/retest research strategy if it validates.

## Critical files
`strategy.py`, `risk.py`, `engine.py`, `models.py`, `config.py`, `config/pilot.json`, `config/paper-policy.json`, `discovery.py`, `market_data.py`, `paper_runner.py`, `broker_contract.py`, `paper_broker.py`, `ibkr.py`, `service_contracts.py`, `replay.py`, `research_validation.py`, `weekly_review.py`.
Reuse: `RiskEngine.round_trip_cost/plan_economics`, `_session_bars`/`_vwap`, `research_strategies.ResearchStrategy`/`StrategyProposal`, `portfolio_runtime.admission_budget`, `diversification_check`, the holdout and cohort logic, the `research-performance` CLI, `broker_drills`.

## Verification
- Plan document reviewed by owner before any ticket starts.
- Future tickets: `python -m unittest discover -s tests -q`, `PYTHONPATH=src:tools python -m unittest discover -s tools/tests -q`, `node --test tests/dashboard_frontend.test.cjs`, `python -m ibkr_day_trader.broker_drills` all stay green. Each strategy change needs a backtest report (in-sample, walk-forward, holdout, 2× cost stress) before paper.
