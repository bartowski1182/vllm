"""Check gate/up row order of a per-expert fused MXFP4 ``gate_up_proj`` export.

Dequantizes one expert from the compressed-tensors checkpoint and compares its
two row halves against the original BF16 ``mlp.experts.gate_up_proj[e]``
(layout ``[2I, K]``, gate rows first in Transformers). ``down_proj`` is compared
the same way as a control for the dequantization itself. Only the needed bytes
of the original shard are fetched (HTTP range reads via HfFileSystem), so no
26 GB download is required.

Usage (inside the vllm-dev container or any env with torch, safetensors,
huggingface_hub):

    python check_muse_gate_up_order.py /models/muse-spark \
        --original meta-models/Muse-Spark-Open-816B-A42B --layer 0 --expert 0
"""

import argparse
import json
import os
import re
import struct

import torch
from safetensors import safe_open

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def dequant_mxfp4(packed: torch.Tensor, scale: torch.Tensor, group: int = 32):
    """packed: uint8 [N, K/2] (low nibble first); scale: uint8 e8m0 [N, K/group]."""
    low = packed & 0x0F
    high = packed >> 4
    nibbles = torch.stack((low, high), dim=-1).reshape(packed.shape[0], -1)
    vals = E2M1[(nibbles & 0x07).long()]
    vals = torch.where(nibbles & 0x08 != 0, -vals, vals)
    n, k = vals.shape
    scales = torch.ldexp(torch.ones_like(scale, dtype=torch.float32), scale.int() - 127)
    return (vals.view(n, k // group, group) * scales[:, :, None]).view(n, k)


def find_keys(weight_map: dict, pattern: str) -> list[str]:
    keys = [k for k in weight_map if re.search(pattern, k)]
    if not keys:
        sample = sorted(k for k in weight_map if ".experts." in k)[:5]
        raise SystemExit(
            f"no tensors matching /{pattern}/ in index; sample expert keys: {sample}"
        )
    return keys


def load_quantized_expert(ckpt_dir: str, layer: int, expert: int, proj: str):
    index = json.load(open(os.path.join(ckpt_dir, "model.safetensors.index.json")))
    pattern = rf"\.layers\.{layer}\.mlp\.experts\.{expert}\.{proj}\.(\w+)$"
    tensors = {}
    for key in find_keys(index["weight_map"], pattern):
        with safe_open(os.path.join(ckpt_dir, index["weight_map"][key]), "pt") as f:
            tensors[re.search(pattern, key).group(1)] = f.get_tensor(key)
    print(f"{key.rsplit('.', 1)[0]}.*: " + ", ".join(
        f"{k}{tuple(v.shape)}/{v.dtype}" for k, v in tensors.items()))
    return dequant_mxfp4(tensors["weight_packed"], tensors["weight_scale"])


def open_original(original: str, filename: str):
    if os.path.isdir(original):
        return open(os.path.join(original, filename), "rb")
    from huggingface_hub import HfFileSystem

    return HfFileSystem().open(f"{original}/{filename}", "rb")


def load_original_expert(original: str, layer: int, expert: int, proj: str):
    with open_original(original, "model.safetensors.index.json") as f:
        index = json.load(f)
    (key,) = find_keys(
        index["weight_map"], rf"\.layers\.{layer}\.mlp\.experts\.{proj}(\.weight)?$"
    )
    shard = index["weight_map"][key]
    with open_original(original, shard) as f:
        (hdr_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hdr_len))
        meta = header[key]
        shape = meta["shape"]
        assert meta["dtype"] == "BF16", meta
        slab = 2 * shape[1] * shape[2]
        f.seek(8 + hdr_len + meta["data_offsets"][0] + expert * slab)
        buf = f.read(slab)
    t = torch.frombuffer(bytearray(buf), dtype=torch.bfloat16).view(shape[1], shape[2])
    print(f"{key}: shape {tuple(shape)} -> expert {expert} slab {tuple(t.shape)}")
    return t.float()


def row_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a, b, dim=1).mean().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt_dir", help="compressed-tensors MXFP4 checkpoint directory")
    p.add_argument("--original", default="meta-models/Muse-Spark-Open-816B-A42B",
                   help="hub repo id or local dir of the BF16 original")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--expert", type=int, default=0)
    args = p.parse_args()

    down_q = load_quantized_expert(args.ckpt_dir, args.layer, args.expert, "down_proj")
    down_o = load_original_expert(args.original, args.layer, args.expert, "down_proj")
    print(f"control  down_proj  cos(dequant, original) = {row_cos(down_q, down_o):.4f}  (expect ~0.98+)")

    gu_q = load_quantized_expert(args.ckpt_dir, args.layer, args.expert, "gate_up_proj")
    gu_o = load_original_expert(args.original, args.layer, args.expert, "gate_up_proj")
    half = gu_o.shape[0] // 2
    gate_first = (row_cos(gu_q[:half], gu_o[:half]) + row_cos(gu_q[half:], gu_o[half:])) / 2
    swapped = (row_cos(gu_q[:half], gu_o[half:]) + row_cos(gu_q[half:], gu_o[:half])) / 2
    interleaved = (row_cos(gu_q[0::2], gu_o[:half]) + row_cos(gu_q[1::2], gu_o[half:])) / 2
    print(f"gate_up_proj  gate-first halves  = {gate_first:.4f}")
    print(f"gate_up_proj  swapped halves     = {swapped:.4f}")
    print(f"gate_up_proj  interleaved rows   = {interleaved:.4f}")
    verdict = max(("gate-first", gate_first), ("swapped", swapped), ("interleaved", interleaved), key=lambda x: x[1])
    print(f"=> checkpoint rows look {verdict[0]} ({verdict[1]:.4f}); vLLM assumes gate-first")


if __name__ == "__main__":
    main()
