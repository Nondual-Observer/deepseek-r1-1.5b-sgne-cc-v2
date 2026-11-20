# Datasets, Base Model, and Licensing

## License Scope (`Apache-2.0`)
The `Apache-2.0` license in [`LICENSE`](LICENSE) applies to the original code, weight-space patch (`patch/FINAL_SGNE_CC_V2_205_PATCH.npz`), patch applier (`patch/apply_sgne_cc_v2_205_to_safetensors.py`), verification scripts (`verification/verify_repo_integrity.py`), generated model output traces (`traces/`), figure cards (`cards/`), and documentation/articles (`README*.md`, `article/`) created by the author.

The source terms below apply to benchmark material wherever it appears, including questions, reference answers and tests copied into the `meta` fields in `traces/*.jsonl`. The author’s Apache-2.0 license covers the author’s own contributions. Each benchmark was selected and repackaged in JSON for this release.

---

## Benchmark Datasets (`datasets/`, 1,054 problems total)

| File | Benchmark | `N` | Upstream Source | Upstream License | Notes |
| :--- | :--- | :---: | :--- | :--- | :--- |
| `datasets/math500_ids_manifest.json` | **MATH-500** | `500` | [`HuggingFaceH4/MATH-500`](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) (the [PRM800K test split](https://github.com/openai/prm800k#math-splits) of Hendrycks MATH) | [PRM800K repository notice](licenses/PRM800K-MIT.txt) and [MATH repository notice](licenses/MATH-MIT.txt). The H4 dataset card gives no separate license for the problem texts. | **Identifiers only.** Raw problem texts and model responses are not redistributed. The manifest lists `idx`, `unique_id`, `subject`, `level` and `prompt_sha256`; per-item scores, token counts and `text_sha256` digests of all `2,500` responses are in [`traces/math500_per_item_results.json`](traces/math500_per_item_results.json). Split: `26` calibration problems + `474` remaining; `421` after also excluding `53` items from the early head-atlas inspection. |
| `datasets/46_dataset_mbpp_120.json` | **MBPP** | `120` | [`google-research-datasets/mbpp`](https://huggingface.co/datasets/google-research-datasets/mbpp) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | First `120` problems from `sanitized/test`; prompt shows the first 3 `assert` checks, while evaluation executes the full `test_list` (`365` assertions total: 3 in 116 tasks, 4 in #58/#223/#226, 5 in #172). |
| `datasets/46_dataset_humaneval_164.json` | **HumanEval** | `164` | [`openai/human-eval`](https://github.com/openai/human-eval/blob/master/LICENSE) | **MIT License** | Complete 164-problem HumanEval benchmark evaluated against hidden `test` unit suites. |
| `datasets/46_dataset_arc_challenge_150.json` | **ARC-Challenge** | `150` | [`allenai/ai2_arc`](https://huggingface.co/datasets/allenai/ai2_arc) | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | 150-problem subset from ARC-Challenge (`test` split). |
| `datasets/46_dataset_bbh_logic_120.json` | **BBH Logic** | `120` | [`suzgunmirac/BIG-Bench-Hard`](https://github.com/suzgunmirac/BIG-Bench-Hard) | **MIT License** | 30 examples each from `logical_deduction_five_objects`, `web_of_lies`, `tracking_shuffled_objects_three_objects` and `boolean_expressions`. |

The publisher of Alcumus raised a [rights claim about the parent Hendrycks competition_math dataset](https://huggingface.co/datasets/huggingface-legal/takedown-notices/blob/6be1fd145590de35cc6c3788a146dcf6a806382a/2025/2025-01-02-AoPS.md). The [H4 card](https://huggingface.co/datasets/HuggingFaceH4/MATH-500/blob/main/README.md) identifies the 500-problem subset. Repository license notices alone do not establish permission to redistribute the underlying problem texts, so this release publishes MATH-500 identifiers, digests and per-item results only. To reproduce the MATH-500 numbers, obtain the problems from the H4 dataset and match them by `unique_id` and `prompt_sha256`.

### Upstream notices

- OpenAI HumanEval — [original MIT notice](licenses/HumanEval-MIT.txt).
- BIG-Bench Hard — [original MIT notice](licenses/BIG-Bench-Hard-MIT.txt). The four tasks come from [BIG-bench](https://github.com/google/BIG-bench).
- Hendrycks MATH and the OpenAI PRM800K split — [MATH notice](licenses/MATH-MIT.txt) and [PRM800K notice](licenses/PRM800K-MIT.txt).

---

## Base Model Weights (Not Redistributed Here)
- **Model:** [`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) (MIT License). The base model card identifies its ancestor as `Qwen2.5-Math-1.5B`, released under Apache-2.0.
- **Base `model.safetensors` SHA-256:** `58858233513d76b8703e72eed6ce16807b523328188e13329257fb9594462945`
