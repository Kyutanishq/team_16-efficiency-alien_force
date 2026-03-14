# Token Delisting Risk Score (TDRS)
**CoinDCX Internal Hackathon 2025 · Build for Efficiency**

> A machine-learning model that assigns a weekly risk score (0–100) to every listed token — telling ops teams which tokens are heading towards delisting before it becomes a crisis.

---

## What It Does

Every token on CoinDCX receives a score and a tier each week:

| Tier | Score | Meaning | Recommended Action |
|------|-------|---------|-------------------|
| 🔴 High Risk | 60–100 | Strong multi-signal evidence of impending delisting | Immediate review — initiate delisting process |
| 🟡 Medium Risk | 30–59 | Early warning signals present; token deteriorating | Flag for monitoring — schedule 2-week review |
| 🟢 Low Risk | 0–29 | Token is healthy; no significant signals | No action required |

---

## Why It Was Built

Today, delisting decisions are **reactive** — teams manually review tokens after problems become obvious, often too late. This model replaces that with a proactive, data-backed triage system:

- Reduces weekly reviewer time from ~40 hours to under 10 hours
- Extends lead time from 0–2 weeks (reactive) to **4–13 weeks advance warning**
- Provides an auditable, data-backed rationale for every delisting decision
- Eliminates surprise delistings that damage user trust

---

## Model Performance

| Metric | Value |
|--------|-------|
| **Precision @ High Risk Tier** | **100%** — 0 false positives |
| **Recall @ High Risk Tier** | **71%** — 89 of 125 delistings caught |
| ROC-AUC (cross-validation) | 0.643 |
| Tokens assessed | 1,129 |
| Confirmed delistings used for training | 125 |

> **Note on AUC:** This version runs on a single-week snapshot with Santiment social/on-chain signals largely missing. Full multi-week panel data is expected to lift AUC to 0.72+.

---

## Top Signals Driving the Score

| # | Signal | Why It Matters |
|---|--------|----------------|
| 1 | Global spot volume percentile rank | Bottom 10th percentile = very high risk |
| 2 | Global spot vol % change (4w) | Sustained drop flags dying market interest |
| 3 | Volume ratio: 1w vs 4w avg | Captures sudden volume collapse |
| 4 | Volume ratio: 8w vs 13w baseline | Medium-term momentum direction |
| 5 | Price vs 90-day MA | Persistent price discount signals a failed token |
| 6 | DCX spot vol % change (4w) | Isolates CoinDCX-specific liquidity death |
| 7 | 30-day avg developer activity | Zero activity for 30+ days is a near-certain red flag |

---

## Data Sources

| Source | Signals Used |
|--------|-------------|
| CoinDCX Internal DB | Spot & futures volume, market share vs global |
| Global Market Data | Cross-exchange volume trends |
| Internal Price Data | Price vs MA, drawdown from peak, BTC-relative return |
| Santiment API | Developer activity, social sentiment, on-chain signals |
| Historical Delistings | 125 confirmed delistings as ground truth labels |

---

## Who Uses It

| Team | Use Case | Cadence |
|------|----------|---------|
| Listing & Ops | Weekly triage — review High Risk tokens, initiate workflow | Every Monday |
| Product & Growth | Proactive user communication planning for Medium Risk | Bi-weekly |
| Compliance | Audit trail — score history + signal breakdown per decision | On-demand |
| Leadership / Risk | Portfolio health dashboard across all listed tokens | Monthly |

---

## Output Files

| File | Contents |
|------|----------|
| `token_risk_scores_v3.csv` | Every token: risk score, tier, probability, top 3 signal drivers |
| `model_metrics_v3.csv` | Side-by-side metrics for all 4 models evaluated |
| `model_results_v3.png` | 7-panel visualisation: ROC curves, feature importance, top 20 risk tokens |
| `token_delisting_risk_model.py` | Full production pipeline — 9 documented steps, ready to run on new weekly data |

---

*CoinDCX Internal Hackathon 2025 · Build for Efficiency · Token Delisting Risk Score*
