# Name That LLM — Nov 1 follow-up

- Submitted kernel `krishnakumarbhat/ntl-recipe-offline` v4 using the required
  `submission.parquet` output (submission ref 57869867).
- Offline kernel completed with no internet. Predictions validated against all
  3 sample IDs and the 18 known class labels. Saved output: `submission_recipe_v4.csv` / `.parquet`.
- The reduced blend used char TF-IDF, word TF-IDF, and the saved Qwen LoRA member;
  grouped OOF balanced accuracy was 0.6702. The public score was **0.375**, rank
  **3067/4156** on the Nov 1 public leaderboard.
- Previous TF-IDF-only submission was 0.308 (rank 3770/4013 at its earlier check).
