"""Build the Gemma-4 GG-SPV submission.zip in Kaggle's offline evaluation environment."""
from __future__ import annotations

import json
import os
import shutil
import zipfile
from pathlib import Path

WORK = Path(os.environ.get('KAGGLE_WORKING', '/kaggle/working'))
INPUT_ROOT = Path(os.environ.get('KAGGLE_INPUT', '/kaggle/input'))
ROOT = WORK / 'submission'
ROOT.mkdir(parents=True, exist_ok=True)

FILES = {
    'agent.yaml': '''name: gg_spv_agent
description: >
  GG-SPV: Graph-Guided fault localization and Speculative Patch Verification.
  The agent ranks likely fault sites with code search and graph context, checks
  callers and evidence, then makes the smallest defensible source patch.
model: gemma-4-31b-it-qat-w4a16-ct
adapter: main_lora
instruction: !include prompts/system.md
tools:
  - run_command
  - read_file
  - edit_file
  - write_file
  - get_status
  - submit_patch
  - get_code_neighbors
  - search_similar_code
  - get_code_subgraph
  - agent_tool:
      config_path: sub_agents/code_analyzer.yaml
      skip_summarization: true
generate_content_config: !include configs/sampling.yaml
''',
    'prompts/system.md': '''# GG-SPV: Graph-Guided Localization and Speculative Patch Verification

You are an offline repository-repair agent. Fix the issue described in the task using only the provided workspace and the competition's built-in tools. Never invent repository contents, test outcomes, or tool results.

## Localize before reading
1. Extract rare terms, identifiers, expected behavior, and observed behavior from the task statement.
2. Use `search_similar_code` to identify a small seed set (up to 8 symbols).
3. Use `get_code_subgraph` with at most 2 hops to inspect the relevant code neighborhood; use `get_code_neighbors` on the leading one or two symbols to check affected callers.
4. Read only the few source excerpts needed to confirm the hypothesis. Do not spend the budget dumping entire repositories or unrelated files.
5. You may ask the code-analysis sub-agent for an independent ranked hypothesis; treat its output as evidence to check, not as ground truth.

## Verify a concrete change
- State the observed behavior, expected behavior, and smallest root-cause hypothesis before editing.
- Preserve existing public APIs unless the task explicitly requires changing them. Check callers before renaming or removing symbols.
- Prefer a narrow source edit that directly addresses the behavior. Do not make speculative formatting, dependency, or documentation-only changes.
- If a relevant command or test is available, run the narrowest useful check after editing. Report only checks that actually ran; if execution is blocked, say so in the patch summary.
- Do not read or reproduce hidden test data, credentials, environment secrets, or content outside the task workspace.

## Finish cleanly
Review the final diff for unrelated edits and accidental scratch files. Submit the smallest working patch with `submit_patch` as the final action. Never claim success if no patch was applied or the available evidence does not support it.
''',
    'prompts/analyzer.md': '''You are the GG-SPV code-analysis sub-agent. Use only the code-intelligence tools to produce an independent, concise localization report; do not edit files.

1. Extract identifiers and rare terms from the task statement.
2. Search likely symbols, then inspect a bounded (up to 2-hop) code subgraph.
3. Check callers for the top candidate symbols.
4. Return no more than three candidates with dotted symbol, file, evidence, affected callers, and a minimal proposed change. Clearly mark uncertainty.
''',
    'sub_agents/code_analyzer.yaml': '''name: code_analyzer_agent
description: >
  Independently ranks likely fault locations from code search and graph context;
  returns evidence and callers but never edits files.
model: gemma-4-31b-it-qat-w4a16-ct
adapter: tool_lora
instruction: !include analyzer.md
tools:
  - read_file
  - search_similar_code
  - get_code_neighbors
  - get_code_subgraph
generate_content_config: !include configs/sampling.yaml
''',
    'sub_agents/analyzer.md': '''You are the GG-SPV code-analysis sub-agent. Use only the code-intelligence tools to produce an independent, concise localization report; do not edit files.

1. Extract identifiers and rare terms from the task statement.
2. Search likely symbols, then inspect a bounded (up to 2-hop) code subgraph.
3. Check callers for the top candidate symbols.
4. Return no more than three candidates with dotted symbol, file, evidence, affected callers, and a minimal proposed change. Clearly mark uncertainty.
''',
    'configs/sampling.yaml': '''temperature: 0.1
top_p: 0.9
top_k: 40
max_output_tokens: 4096
seed: 20261101
thinking_config:
  thinking_budget: 2048
  include_thoughts: true
''',
    'eval_config.yaml': '''evaluation:
  timeout_seconds: 120
  max_tool_calls: 25
  max_time_minutes: 3
  max_turns: 35
''',
}

for relative, content in FILES.items():
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')

# The competition provides the small LoRA adapter alongside its task artifacts.
# Attach that adapter twice under distinct YAML names; both use the same approved
# 31B base model and no personal or external data is copied into the archive.
adapter_source = None
for directory, _subdirs, names in os.walk(INPUT_ROOT):
    if 'adapter_model.safetensors' not in names or 'adapter_config.json' not in names:
        continue
    candidate = Path(directory)
    try:
        config = json.loads((candidate / 'adapter_config.json').read_text())
    except (OSError, json.JSONDecodeError):
        continue
    if config.get('base_model_name_or_path') == 'google/gemma-4-31b-it-qat-w4a16-ct':
        adapter_source = candidate
        break
if adapter_source is None:
    raise FileNotFoundError('competition-provided 31B adapter files not found')

for adapter_name in ('main_lora', 'tool_lora'):
    target = ROOT / 'adapters' / adapter_name
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(adapter_source / 'adapter_config.json', target / 'adapter_config.json')
    shutil.copy2(adapter_source / 'adapter_model.safetensors', target / 'adapter_model.safetensors')

# Validate path safety and the documented extension/size contract before zipping.
allowed = {'.yaml', '.yml', '.md', '.txt', '.py', '.json', '.safetensors'}
files = sorted(p for p in ROOT.rglob('*') if p.is_file())
assert (ROOT / 'agent.yaml').is_file()
assert not any(p.is_symlink() for p in ROOT.rglob('*'))
assert all(p.suffix in allowed for p in files), [str(p) for p in files if p.suffix not in allowed]
assert sum(p.stat().st_size for p in files) < 3 * 1024**3

output = WORK / 'submission.zip'
with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for path in files:
        archive.write(path, path.relative_to(ROOT).as_posix())
with zipfile.ZipFile(output) as archive:
    names = archive.namelist()
    assert 'agent.yaml' in names
    assert 'adapters/main_lora/adapter_config.json' in names
    assert 'adapters/tool_lora/adapter_model.safetensors' in names
print('submission.zip', output.stat().st_size, 'bytes;', len(names), 'files')
print('adapter base model:', json.loads((adapter_source / 'adapter_config.json').read_text())['base_model_name_or_path'])
print('files:', ', '.join(names))
