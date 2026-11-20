#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_repo_integrity.py
Offline integrity and metrics verifier for deepseek-r1-1.5b-sgne-cc-v2.

Requires: numpy.

Checks:
1. Exact 42-entry expected file set in SHA256_MANIFEST.json (rejects missing entries or mismatches).
2. SHA-256 consistency between SHA256_MANIFEST.json and SHA256_AND_METRICS_CERTIFICATE.json.
3. Patch NPZ structure (205 mlp.down_proj columns across L18..L27, shape 1536 x k, unique valid indices).
4. Exact row counts (5,270 total: 2,770 across 20 non-math JSONL traces + 2,500 across 5 MATH-500 runs in
   traces/math500_per_item_results.json), missing-</think> counts, and total_tokens sums against
   trace_summary_counts and benchmarks_5x2 in SHA256_AND_METRICS_CERTIFICATE.json.
"""

import hashlib
import json
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "verification" / "SHA256_MANIFEST.json"
CERT_PATH = REPO_ROOT / "verification" / "SHA256_AND_METRICS_CERTIFICATE.json"

EXPECTED_FILES = {
    "patch/FINAL_SGNE_CC_V2_205_PATCH.npz",
    "patch/apply_sgne_cc_v2_205_to_safetensors.py",
    "datasets/math500_ids_manifest.json",
    "datasets/46_dataset_arc_challenge_150.json",
    "datasets/46_dataset_bbh_logic_120.json",
    "datasets/46_dataset_humaneval_164.json",
    "datasets/46_dataset_mbpp_120.json",
    "cards/01_math__math500_2x2_32k_en.png",
    "cards/01_math__math500_2x2_32k_ru.png",
    "cards/02_code__mbpp120_2x2_32k_en.png",
    "cards/02_code__mbpp120_2x2_32k_ru.png",
    "cards/03_code__humaneval164_2x2_32k_en.png",
    "cards/03_code__humaneval164_2x2_32k_ru.png",
    "cards/04_logic__arc150_2x2_32k_en.png",
    "cards/04_logic__arc150_2x2_32k_ru.png",
    "cards/05_logic__bbh120_2x2_32k_en.png",
    "cards/05_logic__bbh120_2x2_32k_ru.png",
    "cards/06_math__opt_t30_vs_passport_2cand_en.png",
    "cards/06_math__opt_t30_vs_passport_2cand_ru.png",
    "cards/07_logic__arc150_opt_t30_vs_passport_2cand_en.png",
    "cards/07_logic__arc150_opt_t30_vs_passport_2cand_ru.png",
    "traces/math500_per_item_results.json",
    "traces/65_REAL32K_ARC_BASE_T00.jsonl",
    "traces/65_REAL32K_ARC_BASE_T60.jsonl",
    "traces/65_REAL32K_ARC_V2_T00.jsonl",
    "traces/65_REAL32K_ARC_V2_T30.jsonl",
    "traces/65_REAL32K_ARC_V2_T60.jsonl",
    "traces/65_REAL32K_BBH_BASE_T00.jsonl",
    "traces/65_REAL32K_BBH_BASE_T60.jsonl",
    "traces/65_REAL32K_BBH_V2_T00.jsonl",
    "traces/65_REAL32K_BBH_V2_T30.jsonl",
    "traces/65_REAL32K_BBH_V2_T60.jsonl",
    "traces/65_REAL32K_HE_BASE_T00.jsonl",
    "traces/65_REAL32K_HE_BASE_T60.jsonl",
    "traces/65_REAL32K_HE_V2_T00.jsonl",
    "traces/65_REAL32K_HE_V2_T30.jsonl",
    "traces/65_REAL32K_HE_V2_T60.jsonl",
    "traces/65_REAL32K_MBPP_BASE_T00.jsonl",
    "traces/65_REAL32K_MBPP_BASE_T60.jsonl",
    "traces/65_REAL32K_MBPP_V2_T00.jsonl",
    "traces/65_REAL32K_MBPP_V2_T30.jsonl",
    "traces/65_REAL32K_MBPP_V2_T60.jsonl",
}

NON_MATH_BENCH_PREFIX = {
    "MBPP-120": "MBPP",
    "HumanEval-164": "HE",
    "ARC-150": "ARC",
    "BBH-120": "BBH",
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    cert = json.loads(CERT_PATH.read_text(encoding="utf-8"))

    manifest_keys = set(manifest["sha256"].keys())
    assert manifest_keys == EXPECTED_FILES, (
        f"Manifest file set mismatch! missing={sorted(EXPECTED_FILES - manifest_keys)}, "
        f"extra={sorted(manifest_keys - EXPECTED_FILES)}"
    )
    assert set(cert["sha256"].keys()) == EXPECTED_FILES, "Certificate sha256 key set mismatch!"
    assert cert["sha256"] == manifest["sha256"], "Certificate sha256 differs from SHA256_MANIFEST.json!"

    # 1. Verify all 42 SHA-256 checksums
    for rel_path in sorted(EXPECTED_FILES):
        fp = REPO_ROOT / rel_path
        assert fp.exists(), f"Missing file: {rel_path}"
        actual = sha256_file(fp)
        expected = manifest["sha256"][rel_path]
        assert actual == expected, f"SHA-256 mismatch for {rel_path}: {actual} != {expected}"

    # 2. Verify patch structure (205 columns, 314,880 scalars)
    patch_fp = REPO_ROOT / "patch" / "FINAL_SGNE_CC_V2_205_PATCH.npz"
    patch = np.load(str(patch_fp))
    total_cols = 0
    for l in range(28):
        k_idx = f"L{l:02d}_idx"
        if k_idx in patch:
            idxs = patch[k_idx]
            assert len(idxs) == len(set(idxs.tolist())), f"Duplicate indices at layer {l}"
            assert np.all(idxs >= 0) and np.all(idxs < 8960), f"Out-of-range index at layer {l}"
            cols = patch[f"L{l:02d}_wd_cols"]
            assert cols.shape == (1536, len(idxs)), f"Unexpected shape at layer {l}: {cols.shape}"
            total_cols += len(idxs)
    assert total_cols == 205, f"Expected 205 columns, got {total_cols}"
    assert cert["patch"]["total_columns"] == 205, "Certificate total_columns mismatch!"

    # 3. Verify 20 non-math JSONL traces + 5 MATH-500 runs in math500_per_item_results.json
    total_rows = 0
    trace_counts = {}
    for rel_tf in sorted(p for p in EXPECTED_FILES if p.endswith(".jsonl")):
        rows = [json.loads(line) for line in (REPO_ROOT / rel_tf).read_text(encoding="utf-8").splitlines() if line.strip()]
        no_et = sum(1 for r in rows if "</think>" not in (r.get("text") or r.get("response") or ""))
        tot_tok = sum(int(r.get("total_tokens", 0)) for r in rows)
        trace_counts[rel_tf] = {"rows": len(rows), "no_end_think": no_et, "total_tokens_sum": tot_tok}
        total_rows += len(rows)

    m500_res = json.loads((REPO_ROOT / "traces/math500_per_item_results.json").read_text(encoding="utf-8"))
    for tag, mrows in sorted(m500_res.items()):
        no_et = sum(1 for r in mrows if not r["closed_think"])
        tot_tok = sum(int(r["total_tokens"]) for r in mrows)
        trace_counts[f"MATH500_{tag}"] = {"rows": len(mrows), "no_end_think": no_et, "total_tokens_sum": tot_tok}
        total_rows += len(mrows)

    assert total_rows == 5270, f"Expected 5270 total evaluated records, got {total_rows}"
    assert cert.get("trace_summary_counts") == trace_counts, "Certificate trace_summary_counts mismatch!"

    # 4. Cross-check benchmarks_5x2 and POOLED_1054
    b5 = cert["benchmarks_5x2"]
    for bkey, pref in NON_MATH_BENCH_PREFIX.items():
        for tkey in ["T00", "T60"]:
            for arm, atag in [("base", "BASE"), ("sgne_cc", "V2")]:
                rel = f"traces/65_REAL32K_{pref}_{atag}_{tkey}.jsonl"
                assert b5[bkey][tkey][arm]["no_close"] == trace_counts[rel]["no_end_think"]
                assert b5[bkey][tkey][arm]["n"] == trace_counts[rel]["rows"]
    for tkey in ["T00", "T60"]:
        for arm, atag in [("base", "BASE"), ("sgne_cc", "V2")]:
            assert b5["MATH-500"][tkey][arm]["no_close"] == trace_counts[f"MATH500_{atag}_{tkey}"]["no_end_think"]
            assert b5["MATH-500"][tkey][arm]["n"] == trace_counts[f"MATH500_{atag}_{tkey}"]["rows"]

    for tkey in ["T00", "T60"]:
        sum_base_nc = sum(trace_counts[f"traces/65_REAL32K_{pref}_BASE_{tkey}.jsonl"]["no_end_think"] for pref in NON_MATH_BENCH_PREFIX.values()) + trace_counts[f"MATH500_BASE_{tkey}"]["no_end_think"]
        sum_sgne_nc = sum(trace_counts[f"traces/65_REAL32K_{pref}_V2_{tkey}.jsonl"]["no_end_think"] for pref in NON_MATH_BENCH_PREFIX.values()) + trace_counts[f"MATH500_V2_{tkey}"]["no_end_think"]
        sum_base_sol = b5["MATH-500"][tkey]["base"]["solved"] + sum(b5[bkey][tkey]["base"]["solved"] for bkey in NON_MATH_BENCH_PREFIX)
        sum_sgne_sol = b5["MATH-500"][tkey]["sgne_cc"]["solved"] + sum(b5[bkey][tkey]["sgne_cc"]["solved"] for bkey in NON_MATH_BENCH_PREFIX)
        assert b5["POOLED_1054"][tkey]["base_no_close"] == sum_base_nc
        assert b5["POOLED_1054"][tkey]["sgne_no_close"] == sum_sgne_nc
        assert b5["POOLED_1054"][tkey]["base_solved"] == sum_base_sol
        assert b5["POOLED_1054"][tkey]["sgne_solved"] == sum_sgne_sol

    print(f"PASS: {len(EXPECTED_FILES)}/{len(EXPECTED_FILES)} SHA-256 hashes verified.")
    print(f"PASS: Patch structure verified ({total_cols} columns = {total_cols * 1536:,} weights).")
    print(f"PASS: 20 non-math JSONL traces + 5 MATH-500 per-item summaries verified ({total_rows:,} total records match certificate).")


if __name__ == "__main__":
    main()
