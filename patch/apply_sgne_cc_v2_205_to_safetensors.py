#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apply_sgne_cc_v2_205_to_safetensors.py
Standalone reproducible weight-space patch applier for DeepSeek-R1-Distill-Qwen-1.5B (BF16).

Applies `FINAL_SGNE_CC_V2_205_PATCH.npz` (205 native SwiGLU mlp.down_proj columns across layers L18..L27:
135 newline-boundary columns + 70 punctuation columns, representing 314,880 / 1,777,088,000 = 0.017719%
of total parameters, strict spherical norm lock kappa_BF16 = 0.99999339, max_dev < 0.03%) directly to a
standard HuggingFace `model.safetensors` checkpoint.

Requires: numpy, torch, safetensors.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

EXPECTED_BASE_SHA256 = "58858233513d76b8703e72eed6ce16807b523328188e13329257fb9594462945"
EXPECTED_PATCH_SHA256 = "6f435a63eb306047fe55353b7a911d32a17dd999efa65f082dd1d7c3779a0b2c"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True, type=Path)
    ap.add_argument("--patch", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--allow-unverified-base", action="store_true", help="Allow non-canonical base safetensors SHA-256")
    args = ap.parse_args()

    base_st = args.base_dir / "model.safetensors"
    if not base_st.exists():
        raise FileNotFoundError(f"Base weights not found: {base_st}")
    if not args.patch.exists():
        raise FileNotFoundError(f"Patch not found: {args.patch}")

    patch_sha = sha256_file(args.patch)
    if patch_sha != EXPECTED_PATCH_SHA256:
        raise ValueError(f"Patch SHA-256 mismatch: {patch_sha} != {EXPECTED_PATCH_SHA256}")

    base_sha = sha256_file(base_st)
    if base_sha != EXPECTED_BASE_SHA256 and not args.allow_unverified_base:
        raise ValueError(f"Base model.safetensors SHA-256 mismatch: {base_sha} != {EXPECTED_BASE_SHA256}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for fn in ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        src = args.base_dir / fn
        if src.exists():
            shutil.copy2(src, args.out_dir / fn)

    tensors = load_file(str(base_st), device="cpu")
    patch = np.load(str(args.patch))

    kappas = []
    per_layer = {}
    total_neurons = 0
    for l in range(28):
        k_idx = f"L{l:02d}_idx"
        if k_idx not in patch:
            continue
        idxs = patch[k_idx].astype(np.int64)
        if len(idxs) != len(set(idxs.tolist())):
            raise ValueError(f"Duplicate column indices at layer L{l:02d}")
        wkey = f"model.layers.{l}.mlp.down_proj.weight"
        n_cols = tensors[wkey].shape[1]
        if np.any(idxs < 0) or np.any(idxs >= n_cols):
            raise ValueError(f"Out-of-range column index at layer L{l:02d} (0 <= idx < {n_cols})")

        cols = torch.from_numpy(patch[f"L{l:02d}_wd_cols"]).to(torch.bfloat16)
        if cols.shape != (tensors[wkey].shape[0], len(idxs)):
            raise ValueError(f"Shape mismatch at layer L{l:02d}: {cols.shape}")

        orig_norm = tensors[wkey][:, idxs].float().norm(dim=0)
        tensors[wkey][:, idxs] = cols
        new_norm = tensors[wkey][:, idxs].float().norm(dim=0)
        ratios = (new_norm / orig_norm.clamp(min=1e-12)).tolist()
        kappas.extend(ratios)
        per_layer[f"L{l:02d}"] = {"count": len(idxs), "indices": idxs.tolist()}
        total_neurons += len(idxs)

    assert total_neurons == 205, f"Expected 205 edited neurons, got {total_neurons}"

    out_st = args.out_dir / "model.safetensors"
    save_file(tensors, str(out_st), metadata={"format": "pt"})

    cert = {
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "method": "SGNE-CC v2",
        "base_safetensors_sha256": base_sha,
        "patch_file": args.patch.name,
        "patch_sha256": patch_sha,
        "output_safetensors_sha256": sha256_file(out_st),
        "edited_neurons": total_neurons,
        "edited_weights": total_neurons * 1536,
        "edited_fraction_pct": round(total_neurons * 1536 / 1_777_088_000 * 100.0, 6),
        "kappa_bf16_mean": float(np.mean(kappas)),
        "kappa_bf16_max_dev": float(np.max(np.abs(np.array(kappas) - 1.0))),
        "per_layer": per_layer,
    }
    (args.out_dir / "patch_certificate.json").write_text(json.dumps(cert, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in cert.items() if k != "per_layer"}, indent=2))


if __name__ == "__main__":
    main()
