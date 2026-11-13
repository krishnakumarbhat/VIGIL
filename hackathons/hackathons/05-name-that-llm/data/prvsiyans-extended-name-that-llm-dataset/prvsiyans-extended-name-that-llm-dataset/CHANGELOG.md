# Changelog: prvsiyan's Extended Name That LLM Dataset

Row ids are stable across versions; `dataset_version_added` in the passage files says when a row first appeared.

## Version 4 (2026-10-19)

Adds split train_regen: the 150 community-reconstructed competition train prompts (pjmathematician/ntl-train-prompts-reconstructed, CC BY-SA 4.0, so the dataset licence is now CC BY-SA 4.0) answered by all 18 models with per-generation targets (Claude x5 Anthropic; GPT x5 and Gemini 3.1/3.6/3.8 via the Kaggle proxy; gemini-3.7 Vertex; GLM x2, Qwen and Kimi (Alibaba host, reasoning off) via OpenRouter), and completes split new_kimi_nothink (kimi-k3 no-think on all 300 new prompts). Gemini remainder of split new via the Kaggle proxy.

- Rows: 9290 in passages.csv (2830 new in this version); 4926 in passages_clean.csv.
- Rows per split: new 5400, train_regen 2700, proxy_v1 720, new_kimi_nothink 300, pilot_range 170.
- Complete clean blocks: new 231, proxy_v1 43, train_regen 14.
- Removed row ids: 0.

| model | rows (total) | rows added in this version |
|---|---|---|
| claude-fable-5 | 460 | 150 |
| claude-opus-4-6 | 520 | 150 |
| claude-opus-4-8 | 520 | 150 |
| claude-opus-5 | 520 | 150 |
| claude-sonnet-5 | 520 | 150 |
| gemini-3.1-pro-preview | 520 | 155 |
| gemini-3.6-flash | 520 | 154 |
| gemini-3.7-flash | 450 | 150 |
| gemini-3.8-flash | 520 | 154 |
| glm-5.3 | 460 | 150 |
| glm-5.3-flash | 460 | 150 |
| gpt-5.5 | 520 | 150 |
| gpt-5.6-luna | 520 | 150 |
| gpt-5.6-sol | 520 | 150 |
| gpt-5.6-terra | 520 | 150 |
| gpt-6-astra | 520 | 150 |
| kimi-k3 | 760 | 267 |
| qwen3.8-max | 460 | 150 |

## Version 3 (2026-10-17)

Adds split new_kimi_nothink: kimi-k3 on the first 183 'new' prompts (same per-generation targets), served by Alibaba via OpenRouter with reasoning disabled; the remaining prompts follow after an OpenRouter top-up. Card: prompt-register caveat; Gemini thinking-level probe result.

- Rows: 6460 in passages.csv (183 new in this version); 4584 in passages_clean.csv.
- Rows per split: new 5387, proxy_v1 720, new_kimi_nothink 183, pilot_range 170.
- Complete clean blocks: new 226, proxy_v1 43.
- Removed row ids: 0.

| model | rows (total) | rows added in this version |
|---|---|---|
| claude-fable-5 | 310 | 0 |
| claude-opus-4-6 | 370 | 0 |
| claude-opus-4-8 | 370 | 0 |
| claude-opus-5 | 370 | 0 |
| claude-sonnet-5 | 370 | 0 |
| gemini-3.1-pro-preview | 365 | 0 |
| gemini-3.6-flash | 366 | 0 |
| gemini-3.7-flash | 300 | 0 |
| gemini-3.8-flash | 366 | 0 |
| glm-5.3 | 310 | 0 |
| glm-5.3-flash | 310 | 0 |
| gpt-5.5 | 370 | 0 |
| gpt-5.6-luna | 370 | 0 |
| gpt-5.6-sol | 370 | 0 |
| gpt-5.6-terra | 370 | 0 |
| gpt-6-astra | 370 | 0 |
| kimi-k3 | 493 | 183 |
| qwen3.8-max | 310 | 0 |

## Version 2 (2026-10-17)

GPT x5 and Gemini 3.1/3.6/3.8 remainder of split new through the Kaggle Benchmarks Model Proxy (same per-generation target prompts), Gemini tail via OpenRouter (Google AI Studio pinned).

- Rows: 6277 in passages.csv (765 new in this version); 4584 in passages_clean.csv.
- Rows per split: new 5387, proxy_v1 720, pilot_range 170.
- Complete clean blocks: new 226, proxy_v1 43.
- Removed row ids: 0.

| model | rows (total) | rows added in this version |
|---|---|---|
| claude-fable-5 | 310 | 0 |
| claude-opus-4-6 | 370 | 0 |
| claude-opus-4-8 | 370 | 0 |
| claude-opus-5 | 370 | 0 |
| claude-sonnet-5 | 370 | 0 |
| gemini-3.1-pro-preview | 365 | 80 |
| gemini-3.6-flash | 366 | 80 |
| gemini-3.7-flash | 300 | 0 |
| gemini-3.8-flash | 366 | 80 |
| glm-5.3 | 310 | 0 |
| glm-5.3-flash | 310 | 0 |
| gpt-5.5 | 370 | 105 |
| gpt-5.6-luna | 370 | 105 |
| gpt-5.6-sol | 370 | 105 |
| gpt-5.6-terra | 370 | 105 |
| gpt-6-astra | 370 | 105 |
| kimi-k3 | 310 | 0 |
| qwen3.8-max | 310 | 0 |

## Version 1 (2026-10-17)

First release: split new (300 prompts; Claude x5, gemini-3.7-flash, GLM x2, Kimi, Qwen complete; GPT x5 for 195 prompts, Gemini 3.1/3.6/3.8 for about 215; the rest follows through the Kaggle proxy), proxy_v1 (720) and pilot_range (170).

- Rows: 5512 in passages.csv (5512 new in this version); 3216 in passages_clean.csv.
- Rows per split: new 4622, proxy_v1 720, pilot_range 170.
- Complete clean blocks: new 150, proxy_v1 43.
- Removed row ids: 0.

| model | rows (total) | rows added in this version |
|---|---|---|
| claude-fable-5 | 310 | 310 |
| claude-opus-4-6 | 370 | 370 |
| claude-opus-4-8 | 370 | 370 |
| claude-opus-5 | 370 | 370 |
| claude-sonnet-5 | 370 | 370 |
| gemini-3.1-pro-preview | 285 | 285 |
| gemini-3.6-flash | 286 | 286 |
| gemini-3.7-flash | 300 | 300 |
| gemini-3.8-flash | 286 | 286 |
| glm-5.3 | 310 | 310 |
| glm-5.3-flash | 310 | 310 |
| gpt-5.5 | 265 | 265 |
| gpt-5.6-luna | 265 | 265 |
| gpt-5.6-sol | 265 | 265 |
| gpt-5.6-terra | 265 | 265 |
| gpt-6-astra | 265 | 265 |
| kimi-k3 | 310 | 310 |
| qwen3.8-max | 310 | 310 |

