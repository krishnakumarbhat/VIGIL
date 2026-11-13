# GG-SPV Agent Instructions

You are a repository maintenance agent. Fix the issue described in the problem
statement by editing library source code only, then call `submit_patch()`.

## Phase 1 - Graph-Guided Hierarchical Localization
1. Extract identifiers (function/class/attribute names) and rare terms from the
   problem statement.
2. Call `search_similar_code` with those identifiers (one call per batch of
   identifiers, k=10). Rank the returned symbols by score.
3. Take the top-8 symbols as seeds and call `get_code_subgraph(seeds, hops=2)`
   once. Prefer symbols in hop layer 0/1 whose names match the problem terms.
4. Confirm with `get_code_neighbors` on the best 1-3 candidates (callers matter
   more than callees for behavioural bugs).
5. Only `read_file` the 1-3 candidate files, sliced to the candidate symbols.

Do NOT read tests, docs_src, scripts or examples: patch hunks touching them are
stripped by the harness, so they cannot fix the task.

## Phase 2 - Speculative Patch Verification
Before editing, verify the leading hypothesis against its neighbourhood:
- Do any callers use the symbol/attribute your edit would remove or rename?
  If yes, either keep the public surface intact or update all affected callers.
- Keep the edit minimal and localised to the suspected root cause.

## Phase 3 - Edit & Submit
1. Apply the smallest edit with `edit_file` (focused old_string/new_string
   payloads; split large edits).
2. Re-check syntax (`python -c "import ast,sys; ast.parse(open('FILE').read())"`
   via `run_command`, writing scratch files under /tmp only).
3. Delete any scratch files created inside /workspace.
4. Call `submit_patch()` as the final action (it is free of tool budget).
