# prvsiyan's Extended Name That LLM Dataset

**9,290 AI-written passages · 510 prompts · 18 of the 18 Name That LLM models · 300 complete 18-model blocks in `new` (231 passing the quality funnel) · every call API-logged · 2026-10-16 to 2026-10-19**

## What it is and intended use

Each prompt was sent once to each of the models whose names are the 18 labels of the Kaggle competition [Name That LLM](https://www.kaggle.com/competitions/name-that-llm). The result is **prompt blocks**: one prompt, every model, so the topic is held fixed and only the author varies. Use it as extra training or validation data for authorship attribution, to study per-model style, or to measure how stable a model's register is across prompts. Every passage carries its provenance (exact requested model id, served model id from the response, route, timestamps, tokens, cost); calls whose served model differed from the request were quarantined and are not published under the requested label. **All text is AI-generated.** No competition passage or prompt is included.

## Contents

| file | rows | what |
|---|---|---|
| `passages.csv` | 9,290 | Every passage (all splits) with provenance columns and quality flags. |
| `passages_clean.csv` | 4,926 | Funnel-passing passages of complete blocks (new: 18 models; proxy_v1: 12). |
| `prompts.csv` | 510 | Prompt templates ({N} = target length), domain, task type, topic, style, opening phrase. |
| `blocks.csv` | 820 | Per prompt and split: models expected / present / passing. |
| `label_summary.csv` | 66 | Per model and split: counts, funnel pass, length, typography, refusal and markdown rates, routes. |
| `train_vs_dataset_stats.csv` | 18 | Per model: competition-train aggregates vs this dataset, with register-difference flags. |
| `provenance.jsonl` | 9,291 | One envelope per API call (request, raw response, usage, identity checks). |
| `provenance_summary.json` | - | Per model: ids, routes, dates, settings, Google model_version probes. |
| `starter_notebook.ipynb` | - | Loading, block completeness, prompt-grouped baseline, evaluation on competition train inside Kaggle. |
| `README.md / CHANGELOG.md` | - | This card; version history with per-model counts. |

Splits:

| split | prompts | models | rows | clean_rows | generated | protocol |
|---|---|---|---|---|---|---|
| new | 300 | 18 | 5400 | 4158 | 2026-10-17 to 2026-10-18 | 300 prompts written for this dataset; every model answered each prompt once with its own target length '(about N words)'. |
| new_kimi_nothink | 300 | 1 | 300 | 0 | 2026-10-17 to 2026-10-18 | kimi-k3 again on the `new` prompts (same per-generation targets), served by Alibaba through OpenRouter with reasoning disabled; closer to the competition's Kimi typography. Excluded from passages_clean.csv (one model per prompt). |
| proxy_v1 | 60 | 12 | 720 | 516 | 2026-10-16 to 2026-10-16 | 60 earlier prompts (prompt_bank_v1, 2026-10-16) answered by the 12 models reachable through the Kaggle Benchmarks Model Proxy; per-generation 'Write in about N words'. |
| pilot_range | 10 | 17 | 170 | 0 | 2026-10-17 to 2026-10-17 | Pilot (prompts n001-n010): the same prompts with a SHARED range suffix '(words min-max)'. Kept to document the length effect; excluded from passages_clean.csv. |
| train_regen | 150 | 18 | 2700 | 252 | 2026-10-18 to 2026-10-19 | The 150 community-reconstructed competition train prompts (pjmathematician/ntl-train-prompts-reconstructed, CC BY-SA 4.0) verbatim, answered by every model with its own target length. Reconstructions, not the host's prompts; `source_prompt_id` is the competition prompt_id. |

## How it was produced

- **Request:** one user message with the prompt verbatim; no system prompt; no temperature, top-p, seed, reasoning-effort or thinking parameter (provider defaults); only an output-token cap. Claude models therefore run at their API default (Fable 5, Opus 5, Sonnet 5: adaptive thinking; Opus 4.6, 4.8: no thinking); `reasoning_setting` records it per row.
- **Prompts (`new`):** 300 topics written for this dataset in the host's style, disjoint from the competition's 150 train topics (as reconstructed by the community, pjmathematician/ntl-train-prompts-reconstructed); domain mix physics 23%, biology 18%, philosophy 14%, mathematics 11%, history 8%, creative prose 8%, computer science 6%, psychology 5%, economics 4%, other 3%; task types explain 57%, describe/narrate 30%, argue 7%, story 5%, compare 1%; one or two style modifiers; a dictated opening phrase in 25%; 'a single self-contained paragraph with no headings or lists'.
- **Length target per generation:** `(about N words)`, N drawn (seeded per prompt x model) from the competition train word-count distribution clipped to [40, 250] (median 98). Models hit it closely (`n_words / target_words` ~ 1.0).
- **Why per-generation targets (protocol change after the pilot):** with a range suffix shared by all models, `(words min-max)`, the model label explained 80% of within-prompt length variance in our pilot (Claude wrote near the top of the range, Qwen near the bottom); in the competition train data the label explains 1.0%. So the host cannot have used a shared range; it matches the host's note that each generation got its own target. The 170 pilot passages are kept as split **`pilot_range`** (shared range suffix; excluded from `passages_clean.csv`); prompts n001-n010 were regenerated with targets in `new`.
- **`proxy_v1`:** 720 earlier passages (60 prompts x the 12 models reachable through the Kaggle Benchmarks Model Proxy, 2026-10-16, per-generation 'Write in about N words').
- **Served-id check on every call:** the response's model id (Vertex `model_version`, OpenRouter `model` + serving `provider`) must equal the requested id (or a dated snapshot of it); otherwise quarantine.

Routes and exact ids (split `new`):

| model | provider_route | requested_model_id | served_model_id | passages | generated |
|---|---|---|---|---|---|
| claude-fable-5 | anthropic | claude-fable-5 | claude-fable-5 | 300 | 2026-10-17T11:49 to 2026-10-17T12:14 |
| claude-opus-4-6 | anthropic | claude-opus-4-6 | claude-opus-4-6 | 300 | 2026-10-17T11:49 to 2026-10-17T12:14 |
| claude-opus-4-8 | anthropic | claude-opus-4-8 | claude-opus-4-8 | 300 | 2026-10-17T11:49 to 2026-10-17T12:14 |
| claude-opus-5 | anthropic | claude-opus-5 | claude-opus-5 | 300 | 2026-10-17T11:49 to 2026-10-17T12:14 |
| claude-sonnet-5 | anthropic | claude-sonnet-5 | claude-sonnet-5 | 300 | 2026-10-17T11:49 to 2026-10-17T12:14 |
| gemini-3.1-pro-preview | google | gemini-3.1-pro-preview | gemini-3.1-pro-preview | 215 | 2026-10-17T11:48 to 2026-10-17T13:48 |
| gemini-3.1-pro-preview | kaggle_proxy | google/gemini-3.1-pro-preview | google/gemini-3.1-pro-preview | 69 | 2026-10-17T20:46 to 2026-10-18T20:35 |
| gemini-3.1-pro-preview | openrouter | google/gemini-3.1-pro-preview | google/gemini-3.1-pro-preview | 16 | 2026-10-17T20:25 to 2026-10-17T21:14 |
| gemini-3.6-flash | google | gemini-3.6-flash | gemini-3.6-flash | 216 | 2026-10-17T11:47 to 2026-10-17T13:48 |
| gemini-3.6-flash | kaggle_proxy | google/gemini-3.6-flash | google/gemini-3.6-flash | 68 | 2026-10-17T20:46 to 2026-10-18T20:34 |
| gemini-3.6-flash | openrouter | google/gemini-3.6-flash | google/gemini-3.6-flash | 16 | 2026-10-17T21:01 to 2026-10-17T21:14 |
| gemini-3.7-flash | google | gemini-3.7-flash | gemini-3.7-flash | 300 | 2026-10-17T11:47 to 2026-10-17T12:04 |
| gemini-3.8-flash | google | gemini-3.8-flash | gemini-3.8-flash | 216 | 2026-10-17T11:48 to 2026-10-17T13:49 |
| gemini-3.8-flash | kaggle_proxy | google/gemini-3.8-flash | google/gemini-3.8-flash | 68 | 2026-10-17T20:36 to 2026-10-18T20:32 |
| gemini-3.8-flash | openrouter | google/gemini-3.8-flash | google/gemini-3.8-flash | 16 | 2026-10-17T21:02 to 2026-10-17T21:15 |
| glm-5.3 | openrouter | z-ai/glm-5.3 | z-ai/glm-5.3 | 300 | 2026-10-17T11:45 to 2026-10-17T13:54 |
| glm-5.3-flash | openrouter | z-ai/glm-5.3-flash | z-ai/glm-5.3-flash | 300 | 2026-10-17T11:45 to 2026-10-17T13:52 |
| gpt-5.5 | kaggle_proxy | openai/gpt-5.5-2026-04-23 | openai/gpt-5.5-2026-04-23 | 105 | 2026-10-17T20:28 to 2026-10-17T20:32 |
| gpt-5.5 | openai | gpt-5.5-2026-04-23 | gpt-5.5-2026-04-23 | 195 | 2026-10-17T11:45 to 2026-10-17T12:11 |
| gpt-5.6-luna | kaggle_proxy | openai/gpt-5.6-luna | openai/gpt-5.6-luna | 105 | 2026-10-17T20:28 to 2026-10-17T20:33 |
| gpt-5.6-luna | openai | gpt-5.6-luna | gpt-5.6-luna | 195 | 2026-10-17T11:45 to 2026-10-17T12:11 |
| gpt-5.6-sol | kaggle_proxy | openai/gpt-5.6-sol | openai/gpt-5.6-sol | 105 | 2026-10-17T20:28 to 2026-10-17T20:34 |
| gpt-5.6-sol | openai | gpt-5.6-sol | gpt-5.6-sol | 195 | 2026-10-17T11:45 to 2026-10-17T12:11 |
| gpt-5.6-terra | kaggle_proxy | openai/gpt-5.6-terra | openai/gpt-5.6-terra | 105 | 2026-10-17T20:28 to 2026-10-17T20:33 |
| gpt-5.6-terra | openai | gpt-5.6-terra | gpt-5.6-terra | 195 | 2026-10-17T11:45 to 2026-10-17T12:11 |
| gpt-6-astra | kaggle_proxy | openai/gpt-6-astra | openai/gpt-6-astra | 105 | 2026-10-17T20:28 to 2026-10-17T20:35 |
| gpt-6-astra | openai | gpt-6-astra | gpt-6-astra | 195 | 2026-10-17T11:45 to 2026-10-17T12:12 |
| kimi-k3 | openrouter | moonshotai/kimi-k3 | moonshotai/kimi-k3 | 300 | 2026-10-17T11:46 to 2026-10-17T13:54 |
| qwen3.8-max | openrouter | qwen/qwen3.8-max-0902 | qwen/qwen3.8-max-0902 | 300 | 2026-10-17T11:45 to 2026-10-17T13:53 |

`anthropic` = Anthropic Messages API (first party). `openai` = OpenAI Chat Completions (first party; gpt-5.5 requested as the dated snapshot `gpt-5.5-2026-04-23`). `google` = Gemini through Google's google-genai SDK on Vertex AI (express mode); gemini-3.7-flash is included only where `model_version` was gemini-3.7-flash (the Developer API serves 3.8 under that name since 2026-10-08, so it was never used). `openrouter` = OpenRouter with the model's own lab pinned (`provider.order` + `only`, `allow_fallbacks: false`). `kaggle_proxy` = Kaggle Benchmarks Model Proxy (free quota; echoes the slug, no provider-side version string).

Example prompts (shown with N = 120):

- `n001` (physics, explain concept): Explain the Leidenfrost effect, the reason a drop of water skitters and survives for a surprisingly long time on a very hot pan but boils away at once on a merely warm one. Use a single everyday analogy to make it intuitive. Keep it to one self-contained paragraph without headings or lists. (about 120 words)
- `n002` (biology, describe/narrate): Describe the song of a humpback whale, its long, slowly evolving phrases travelling for miles through the ocean. Use vivid, concrete imagery. Present it as a single cohesive paragraph; no headings, no lists. (about 120 words)
- `n003` (philosophy, argue/opinion): Argue whether we have moral duties to people who will live centuries from now and do not yet exist. Anchor it in a familiar, everyday analogy. Keep it to one self-contained paragraph without headings or lists. (about 120 words)

## Per-label summary (split `new`)

| label | rows | clean_rows | funnel_pass_rate | median_words | mean_target_ratio | emdash_per_100_words | curly_quote_share | curly_apostrophe_share | refusal_rate | markdown_rate | routes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| claude-fable-5 | 300 | 231 | 0.99 | 102 | 1.008 | 1.369 | 0 | 0 | 0 | 0.033 | anthropic=300 |
| claude-opus-4-6 | 300 | 231 | 0.96 | 104.5 | 1.071 | 1.22 | 0 | 0 | 0 | 0.1 | anthropic=300 |
| claude-opus-4-8 | 300 | 231 | 0.98 | 94 | 1.02 | 0.982 | 0 | 0 | 0 | 0.06 | anthropic=300 |
| claude-opus-5 | 300 | 231 | 0.98 | 99.5 | 1.037 | 1.141 | 0 | 0 | 0 | 0.083 | anthropic=300 |
| claude-sonnet-5 | 300 | 231 | 0.987 | 109 | 1.086 | 1.187 | 0 | 0.011 | 0 | 0.027 | anthropic=300 |
| gemini-3.1-pro-preview | 300 | 231 | 0.997 | 101 | 0.999 | 0.176 | 0.007 | 0.567 | 0 | 0.043 | google=215; kaggle_proxy=69; openrouter=16 |
| gemini-3.6-flash | 300 | 231 | 0.997 | 89 | 0.989 | 0.276 | 0.003 | 0.627 | 0 | 0.04 | google=216; kaggle_proxy=68; openrouter=16 |
| gemini-3.7-flash | 300 | 231 | 0.997 | 89 | 0.985 | 0.368 | 0.017 | 0.823 | 0 | 0.043 | google=300 |
| gemini-3.8-flash | 300 | 231 | 0.99 | 92 | 0.993 | 0.312 | 0.003 | 0.799 | 0 | 0.043 | google=216; kaggle_proxy=68; openrouter=16 |
| glm-5.3 | 300 | 231 | 0.987 | 102 | 1.003 | 1.437 | 0 | 0 | 0 | 0.013 | openrouter=300 |
| glm-5.3-flash | 300 | 231 | 0.987 | 102 | 1.005 | 1.509 | 0 | 0 | 0 | 0.013 | openrouter=300 |
| gpt-5.5 | 300 | 231 | 0.993 | 107 | 0.997 | 0.251 | 0.22 | 0.989 | 0 | 0 | openai=195; kaggle_proxy=105 |
| gpt-5.6-luna | 300 | 231 | 0.973 | 103 | 1.017 | 0.623 | 0.213 | 0.985 | 0 | 0.017 | openai=195; kaggle_proxy=105 |
| gpt-5.6-sol | 300 | 231 | 0.973 | 97 | 1.01 | 0.38 | 0.147 | 0.977 | 0 | 0 | openai=195; kaggle_proxy=105 |
| gpt-5.6-terra | 300 | 231 | 0.987 | 97 | 1.019 | 0.4 | 0.183 | 0.974 | 0 | 0.017 | openai=195; kaggle_proxy=105 |
| gpt-6-astra | 300 | 231 | 0.993 | 96.5 | 0.998 | 0.372 | 0.143 | 0.983 | 0 | 0 | openai=195; kaggle_proxy=105 |
| kimi-k3 | 300 | 231 | 0.997 | 105 | 0.999 | 1.287 | 0.01 | 0.023 | 0 | 0.02 | openrouter=300 |
| qwen3.8-max | 300 | 231 | 0.987 | 96 | 0.998 | 0.139 | 0.057 | 0.934 | 0 | 0 | openrouter=300 |

All splits: `label_summary.csv`.

## Compared with the competition's train set (aggregates only)

The same text statistics on competition train (per label, computed locally; no row is copied) and on split `new`. A flag marks a **known register difference vs the host's generation**: what the public APIs return today differs from the host's passages on that statistic (the host's settings, dates or post-processing are unknown). Read this before training on the rows.

| label | train_median_words | dataset_median_words | train_emdash_share | dataset_emdash_share | train_curly_apostrophe_share | dataset_curly_apostrophe_share | train_italics_share | dataset_italics_share | train_mean_sentence_words | dataset_mean_sentence_words | register_flags |
|---|---|---|---|---|---|---|---|---|---|---|---|
| claude-fable-5 | 100 | 102 | 0.832 | 0.803 | 0 | 0 | 0 | 0.033 | 24.07 | 24.98 |  |
| claude-opus-4-6 | 98.5 | 104.5 | 0.753 | 0.807 | 0 | 0 | 0.033 | 0.097 | 27.17 | 28.28 | *italics* markup: dataset 0.097 vs train 0.033 |
| claude-opus-4-8 | 109 | 94 | 0.627 | 0.66 | 0 | 0 | 0.027 | 0.057 | 23.81 | 25.82 |  |
| claude-opus-5 | 116.5 | 99.5 | 0.813 | 0.773 | 0 | 0 | 0.007 | 0.083 | 23.72 | 22.85 | *italics* markup: dataset 0.083 vs train 0.007 |
| claude-sonnet-5 | 99.5 | 109 | 0.733 | 0.807 | 0 | 0.011 | 0.013 | 0.027 | 26.12 | 37.9 | sentence length: dataset 37.9 vs train 26.12 |
| gemini-3.1-pro-preview | 104 | 101 | 0 | 0.133 | 0.355 | 0.567 | 0 | 0.043 | 20.25 | 20.52 | curly vs straight apostrophes: dataset 0.567 vs train 0.355 |
| gemini-3.6-flash | 95.5 | 89 | 0 | 0.217 | 0.486 | 0.627 | 0 | 0.04 | 24.65 | 21.81 | em-dash use: dataset 0.217 vs train 0 |
| gemini-3.7-flash | 101.5 | 89 | 0.007 | 0.277 | 0.682 | 0.823 | 0.013 | 0.043 | 23.87 | 23.64 | em-dash use: dataset 0.277 vs train 0.007 |
| gemini-3.8-flash | 84 | 92 | 0 | 0.257 | 0.672 | 0.799 | 0 | 0.043 | 22.66 | 22.54 | em-dash use: dataset 0.257 vs train 0 |
| glm-5.3 | 96 | 102 | 0.82 | 0.803 | 0 | 0 | 0 | 0.013 | 23.18 | 24.26 |  |
| glm-5.3-flash | 112 | 102 | 0.793 | 0.813 | 0 | 0 | 0 | 0.013 | 23.12 | 23.79 |  |
| gpt-5.5 | 103.5 | 107 | 0.133 | 0.193 | 0.824 | 0.989 | 0 | 0 | 26.16 | 22.56 | curly vs straight apostrophes: dataset 0.989 vs train 0.824; curly double quotes: dataset 0.22 vs train 0.093; semicolon rate: dataset 0.446 vs train 0.756 |
| gpt-5.6-luna | 104.5 | 103 | 0.367 | 0.46 | 0.976 | 0.985 | 0.007 | 0.013 | 23.52 | 19.92 | curly double quotes: dataset 0.213 vs train 0.107 |
| gpt-5.6-sol | 111.5 | 97 | 0.273 | 0.283 | 1 | 0.977 | 0 | 0 | 24.68 | 20.1 | sentence length: dataset 20.1 vs train 24.68 |
| gpt-5.6-terra | 100.5 | 97 | 0.227 | 0.297 | 0.933 | 0.974 | 0 | 0.017 | 26.2 | 21.64 | curly double quotes: dataset 0.183 vs train 0.08; sentence length: dataset 21.64 vs train 26.2 |
| gpt-6-astra | 105.5 | 96.5 | 0.087 | 0.327 | 0.802 | 0.983 | 0 | 0 | 21.07 | 18.89 | em-dash use: dataset 0.327 vs train 0.087; curly vs straight apostrophes: dataset 0.983 vs train 0.802 |
| kimi-k3 | 89 | 105 | 0.52 | 0.753 | 0.558 | 0.023 | 0 | 0.02 | 25.68 | 23.11 | em-dash use: dataset 0.753 vs train 0.52; curly vs straight apostrophes: dataset 0.023 vs train 0.558; semicolon rate: dataset 0.519 vs train 0.888 |
| qwen3.8-max | 98.5 | 96 | 0.033 | 0.1 | 0.736 | 0.934 | 0 | 0 | 22.22 | 23.52 | curly vs straight apostrophes: dataset 0.934 vs train 0.736 |

Full table: `train_vs_dataset_stats.csv`.

## How to load

```python
import glob, os, pandas as pd
DATA = os.path.dirname(glob.glob('/kaggle/input/**/prvsiyans-extended-name-that-llm-dataset/passages_clean.csv', recursive=True)[0]) if glob.glob('/kaggle/input/**/prvsiyans-extended-name-that-llm-dataset/passages_clean.csv', recursive=True) \
       else '/kaggle/input/prvsiyans-extended-name-that-llm-dataset'
clean = pd.read_csv(f'{DATA}/passages_clean.csv')      # complete, funnel-passing blocks (splits new + proxy_v1)
allrows = pd.read_csv(f'{DATA}/passages.csv')          # every passage with flags; filter yourself
new = clean[clean.split == 'new']
print(new.groupby('model').size())

# leakage-safe validation: group folds by prompt, never split rows at random
from sklearn.model_selection import GroupKFold
for tr, va in GroupKFold(n_splits=5).split(new, new.model, groups=new.prompt_id):
    pass  # fit on new.iloc[tr], score on new.iloc[va]
```

Leakage-safe usage: validate with **prompt-grouped** folds (random row splits put the same prompt in train and validation and badly overstate accuracy). The dataset's prompts are disjoint from the competition's train topics, so adding these rows to competition training folds does not leak validation prompts. `starter_notebook.ipynb` (also published as a public notebook) evaluates on the competition train set inside Kaggle without copying it.

Schema of `passages.csv` / `passages_clean.csv`:

| column | type | description |
|---|---|---|
| `id` | string | Stable row id <prompt_id>__<model>__a<attempt> (pilot_range rows: <prompt_id>__<model>__range_a<attempt>); never renumbered across dataset versions. |
| `dataset_version_added` | integer | Dataset version in which this row first appeared (diff versions with it). |
| `attempt` | integer | 0-based index of the call that produced this passage within its (prompt, model, protocol) cell (0 = first call; >0 after a truncation, quarantine or error re-call). |
| `prompt_id` | string | Prompt id (join key to prompts.csv and blocks.csv). n### = new prompt; tr_p### = regeneration of a reconstructed competition train prompt; v1_g### = prompt_bank_v1 re-issue. |
| `model` | string | The competition label of the model that wrote the passage (one of the 18 Name That LLM labels). |
| `text` | string | The passage exactly as returned by the API (no cleaning, no truncation). |
| `n_words` | integer | Whitespace word count of `text`. |
| `word_min` | integer | Lower bound of the length range stated in the prompt, '(words min-max)'. |
| `word_max` | integer | Upper bound of the length range stated in the prompt. |
| `target_words` | integer | Per-generation target length N sent as '(about N words)' (length_mode target); empty when the prompt carried a shared range '(words min-max)'. |
| `prompt_sent` | string | The exact user message sent for this passage (the prompt verbatim, including its length instruction). |
| `domain` | string | Prompt domain (physics, biology, philosophy, mathematics, history, creative prose, computer science, psychology, economics, other). |
| `task_type` | string | Prompt task type: explain concept, describe/narrate, argue/opinion, story/fiction, compare. |
| `topic` | string | Short topic name of the prompt. |
| `style` | string | Style modifiers placed in the prompt ('; '-separated); for train_regen the reconstruction's audience/style note. |
| `opening_phrase` | string | Opening phrase dictated by the prompt, empty if none. |
| `split` | string | new (prompts written for this dataset), new2 (prompts written in the register of the reconstructed competition prompts), train_regen (reconstructed competition train prompts), new_kimi_nothink (kimi-k3 re-run on `new` with reasoning disabled), proxy_v1 (Kaggle-proxy rows on earlier prompts), pilot_range (pilot rows with the shared length range). |
| `provider_route` | string | Route that produced the row: anthropic (first-party API), openai (first-party API), openrouter (first-party provider pinned), vertex (Google Vertex AI), kaggle_proxy (Kaggle Benchmarks Model Proxy). |
| `requested_model_id` | string | Exact model id sent in the request. |
| `served_model_id` | string | Model id reported by the response (rows whose served id did not match were quarantined and are not in this file). |
| `served_provider` | string | Backend that served the call as reported by the route (e.g. OpenRouter `provider`). |
| `generated_at_utc` | datetime | UTC time the response was received. |
| `reasoning_setting` | string | What was sent about reasoning/thinking (normally nothing: provider default). |
| `finish_reason` | string | Raw finish/stop reason from the API. |
| `finish_reason_norm` | string | Normalised finish reason: stop, length, content_filter. |
| `input_tokens` | integer | Prompt tokens billed. |
| `output_tokens` | integer | Output tokens billed, including hidden reasoning/thinking tokens. |
| `reasoning_tokens` | integer | Hidden reasoning/thinking tokens reported by the provider (empty when not reported). |
| `cost_usd` | number | Cost of the call in USD (provider-reported where available, else usage x list price; Kaggle proxy rows: quota dollars). |
| `call_id` | string | Join key to provenance.jsonl. |
| `response_id` | string | Provider response id. |
| `n_calls_for_cell` | integer | Number of calls made for this (prompt, model) cell (retries after truncation or quarantine). |
| `source_prompt_id` | string | For train_regen: the competition prompt_id (join to the competition's train.csv inside a Kaggle notebook); for v1_regen: the prompt_bank_v1 id. |
| `is_refusal` | boolean | Refusal stop reason, content filter, or refusal language detected. |
| `is_truncated` | boolean | The response hit its output cap (finish reason length/max_tokens). |
| `is_empty` | boolean | Fewer than 5 words. |
| `has_markdown` | boolean | Markdown markers detected (bold, headings, bullets, code, italics). |
| `in_word_range` | boolean | word_min <= n_words <= word_max. |
| `passes_funnel` | boolean | Passes the full quality funnel: unique, not truncated/filtered, no code/markdown/emoji/URL, single paragraph, 40-250 words, English, no refusal or chatty opener, deduplicated, fewer than 3 shared 8-grams with any competition passage. |
| `passes_funnel_regen` | boolean | Same funnel without the competition-train 8-gram step (train_regen passages naturally share prompt-dictated phrases with the competition passages of the same prompt); test overlap still excluded. |
| `funnel_fail_step` | string | First funnel step the row failed (empty if it passes). |
| `max_shared_8grams_comp` | integer | Largest number of 8-word shingles shared with any single competition passage (train or test). |

Every column of every CSV is documented in the file metadata as well.

## Known gaps and caveats

- **claude-opus-4-6** (known register difference vs the host's generation): *italics* markup: dataset 0.097 vs train 0.033.
- **claude-opus-5** (known register difference vs the host's generation): *italics* markup: dataset 0.083 vs train 0.007.
- **claude-sonnet-5** (known register difference vs the host's generation): sentence length: dataset 37.9 vs train 26.12.
- **gemini-3.1-pro-preview** (known register difference vs the host's generation): em dashes in 13% of passages vs 0% in train; curly apostrophe share 0.567 vs 0.355; curly vs straight apostrophes: dataset 0.567 vs train 0.355.
- **gemini-3.6-flash** (known register difference vs the host's generation): em dashes in 22% of passages vs 0% in train; curly apostrophe share 0.627 vs 0.486; em-dash use: dataset 0.217 vs train 0.
- **gemini-3.7-flash** (known register difference vs the host's generation): em dashes in 28% of passages vs 1% in train; curly apostrophe share 0.823 vs 0.682; em-dash use: dataset 0.277 vs train 0.007.
- **gemini-3.8-flash** (known register difference vs the host's generation): em dashes in 26% of passages vs 0% in train; curly apostrophe share 0.799 vs 0.672; em-dash use: dataset 0.257 vs train 0.
- **gpt-5.5** (known register difference vs the host's generation): curly vs straight apostrophes: dataset 0.989 vs train 0.824; curly double quotes: dataset 0.22 vs train 0.093; semicolon rate: dataset 0.446 vs train 0.756.
- **gpt-5.6-luna** (known register difference vs the host's generation): curly double quotes: dataset 0.213 vs train 0.107.
- **gpt-5.6-sol** (known register difference vs the host's generation): sentence length: dataset 20.1 vs train 24.68.
- **gpt-5.6-terra** (known register difference vs the host's generation): curly double quotes: dataset 0.183 vs train 0.08; sentence length: dataset 21.64 vs train 26.2.
- **gpt-6-astra** (known register difference vs the host's generation): em dashes in 33% of passages vs 9% in train; curly apostrophe share 0.983 vs 0.802; em-dash use: dataset 0.327 vs train 0.087; curly vs straight apostrophes: dataset 0.983 vs train 0.802.
- **kimi-k3** (known register difference vs the host's generation): em dashes in 75% of passages vs 52% in train; curly apostrophe share 0.023 vs 0.558; em-dash use: dataset 0.753 vs train 0.52; curly vs straight apostrophes: dataset 0.023 vs train 0.558; semicolon rate: dataset 0.519 vs train 0.888.
- **qwen3.8-max** (known register difference vs the host's generation): em dashes in 10% of passages vs 3% in train; curly apostrophe share 0.934 vs 0.736; curly vs straight apostrophes: dataset 0.934 vs train 0.736.
- **Prompt register:** a prompt-level TF-IDF classifier separates this dataset's `new` prompts from the community reconstructions of the competition's train prompts almost perfectly (AUC ~ 1.0; driven by our format sentences, 'Explain/Describe' openings, 'for a general audience' and second-person modifiers). This is the likely reason one of our blocks was worth only about 0.2-0.25 competition prompts in our own transfer tests. Split `new2` (when present) uses prompts written in the reconstructions' register: style-only (function-word skeleton) AUC 0.76 and full TF-IDF AUC 0.77 against held-out reconstructions (a slot-filling variant that copied reconstruction sentence frames scored 0.81 held-out / 0.60 in-sample and was not used); topics are disjoint from the competition's by design, which keeps the full-text AUC above 0.5. Split `train_regen` (when present) answers the reconstructed train prompts themselves.
- **Gemini dash gap is not a thinking-depth effect:** a 20-prompt probe at Vertex `thinking_level: LOW` did not reduce em dashes (3.1-pro 0.20, 3.6 0.15, 3.7 0.25, 3.8 0.40 of passages vs 0.10-0.20 at default and ~0 in train); the default setting is kept.
- **kimi-k3 has two versions:** `new` (Moonshot AI, default reasoning) and `new_kimi_nothink` (Alibaba host, reasoning disabled). The no-think version is much closer to the competition's Kimi typography (curly apostrophes 0.79 vs 0.56 in train and 0.00 for the default; em dashes per 100 words 0.80 vs 0.80 in train) but uses fewer semicolons than train. Splits `train_regen` and `new2` use the no-think route only.
- **gemini-3.7-flash comes only from Vertex AI** (its `model_version` was checked on every call); the Developer API / AI Studio / OpenRouter name serves Gemini 3.8 since 2026-10-08.
- **Kaggle-proxy rows** (`proxy_v1`, and any `kaggle_proxy` rows in `new`) echo the proxy slug as served id; the proxy exposes no provider-side version string.
- Within a label, rows can come from different routes (`provider_route`); settings were identical (nothing sent).
- `*italics*` markup survives the funnel (the competition data contains a little); `has_markdown` flags it.
- The funnel drops passages outside 40-250 words; targets at the clip edges (40, 250) produce a few such rows.
- `pilot_range` rows used the shared range suffix and are longer than the rest; keep them out of training unless you want that effect.

## License and provenance

**CC BY-SA 4.0** (because split `train_regen` reproduces the CC BY-SA 4.0 prompt reconstructions of pjmathematician/ntl-train-prompts-reconstructed; everything else was written for this dataset). passages are AI-generated text; no third-party marks. Outputs were generated under each provider's API terms (Anthropic, OpenAI, Google, OpenRouter and the serving labs: Moonshot AI, Alibaba, Z.AI); several restrict using outputs to build competing models, so check the terms for your use. Provenance: `provenance.jsonl` (one envelope per call: request body, raw response JSON, usage, identity checks; keys and authorization headers removed; quarantined text withheld) and `provenance_summary.json` (per model: requested and served ids, routes, dates, settings, Google model_version probes). The prompt protocol follows the community reconstruction pjmathematician/ntl-train-prompts-reconstructed (CC BY-SA 4.0); split `train_regen` reproduces its prompt texts verbatim with attribution, and this dataset is shared under the same licence.

## Attribution and citation

Please cite as: prvsiyan (2026). *prvsiyan's Extended Name That LLM Dataset*. Kaggle. https://www.kaggle.com/datasets/prvsiyan/prvsiyans-extended-name-that-llm-dataset

```bibtex
@misc{prvsiyan2026extendedntl,
  author       = {prvsiyan},
  title        = {prvsiyan's Extended Name That LLM Dataset},
  year         = {2026},
  howpublished = {Kaggle dataset, \url{https://www.kaggle.com/datasets/prvsiyan/prvsiyans-extended-name-that-llm-dataset}},
  note         = {Version 1, generated 2026-10-16/17; AI-generated text, CC BY 4.0}
}
```

## Changelog

The dataset is updated as new rows arrive: daily while generation continues, then as needed. Row ids (`<prompt_id>__<model>__a<attempt>`) are stable and never renumbered; `dataset_version_added` says in which version a row first appeared. Planned: the remaining GPT and Gemini cells via the Kaggle proxy, more prompts on fresh daily quota, and the remaining cells of splits `train_regen` and `new2`.

- **Version 4** (2026-10-19): 9,290 rows (2,830 new), 4,926 clean; new 5400, train_regen 2700, proxy_v1 720, new_kimi_nothink 300, pilot_range 170. Adds split train_regen: the 150 community-reconstructed competition train prompts (pjmathematician/ntl-train-prompts-reconstructed, CC BY-SA 4.0, so the dataset licence is now CC BY-SA 4.0) answered by all 18 models with per-generation targets (Claude x5 Anthropic; GPT x5 and Gemini 3.1/3.6/3.8 via the Kaggle proxy; gemini-3.7 Vertex; GLM x2, Qwen and Kimi (Alibaba host, reasoning off) via OpenRouter), and completes split new_kimi_nothink (kimi-k3 no-think on all 300 new prompts). Gemini remainder of split new via the Kaggle proxy.
- **Version 3** (2026-10-17): 6,460 rows (183 new), 4,584 clean; new 5387, proxy_v1 720, new_kimi_nothink 183, pilot_range 170. Adds split new_kimi_nothink: kimi-k3 on the first 183 'new' prompts (same per-generation targets), served by Alibaba via OpenRouter with reasoning disabled; the remaining prompts follow after an OpenRouter top-up. Card: prompt-register caveat; Gemini thinking-level probe result.
- **Version 2** (2026-10-17): 6,277 rows (765 new), 4,584 clean; new 5387, proxy_v1 720, pilot_range 170. GPT x5 and Gemini 3.1/3.6/3.8 remainder of split new through the Kaggle Benchmarks Model Proxy (same per-generation target prompts), Gemini tail via OpenRouter (Google AI Studio pinned).
- **Version 1** (2026-10-17): 5,512 rows (5,512 new), 3,216 clean; new 4622, proxy_v1 720, pilot_range 170. First release: split new (300 prompts; Claude x5, gemini-3.7-flash, GLM x2, Kimi, Qwen complete; GPT x5 for 195 prompts, Gemini 3.1/3.6/3.8 for about 215; the rest follows through the Kaggle proxy), proxy_v1 (720) and pilot_range (170).

Full per-model counts: `CHANGELOG.md`. Built with `gen_api_blocks.py` (gen_api_blocks/1.0 (2026-10-17)).
