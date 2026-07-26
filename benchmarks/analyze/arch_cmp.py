#!/usr/bin/env python3
"""arch_cmp.py — architecture facts + bandwidth-model test of the 'size vs quantization' confound."""
import json, os

MODELS_DIR = os.environ.get("MODELS_DIR", "/data/incoming")
if not os.path.isdir(MODELS_DIR):
    raise SystemExit(f"MODELS_DIR={MODELS_DIR} not present — this script reads safetensors "
                     "headers, so it only runs on a machine holding the models.")

MODELS = [("Qwen3-8B", "8B BF16"), ("gemma-4-12B-it-qat-w4a16-ct", "12B w4a16"),
          ("gemma-4-31B-it-qat-w4a16-ct", "31B w4a16")]
for d, lbl in MODELS:
    c = json.load(open(os.path.join(MODELS_DIR, d, "config.json")))
    t = c.get("text_config", c)
    q = c.get("quantization_config", {}) or {}
    print(f"{lbl:>10}: layers={t.get('num_hidden_layers')} hidden={t.get('hidden_size')} "
          f"heads={t.get('num_attention_heads')}/{t.get('num_key_value_heads')} "
          f"inter={t.get('intermediate_size')} vocab={t.get('vocab_size')} "
          f"swa={t.get('sliding_window')} quant={q.get('quant_method', 'none')}")

print("\n=== bandwidth model test: T = W/(N*B) + C ===")
print("If TP2's gain came purely from halving bandwidth-bound weight traffic,")
print("the implied per-GPU bandwidth B must be <= hardware peak (800 GB/s = 745 GiB/s).\n")
HW_GIBS = 800 / 1.0737   # 800 GB/s -> GiB/s
for lbl, W, tps1, tps2 in [("8B BF16", 14.02, 46.7, 79.6), ("12B w4a16", 9.56, 50.3, 59.9)]:
    t1, t2 = 1000 / tps1, 1000 / tps2          # ms per token
    dt = t1 - t2
    B = (W / 2) / dt                            # GiB/ms
    print(f"{lbl:>10}: T1={t1:6.2f}ms T2={t2:6.2f}ms  dT={dt:5.2f}ms")
    print(f"{'':>10}  implied per-GPU B = (W/2)/dT = {W/2:.2f}/{dt:.2f} = {B*1000:7.0f} GiB/s "
          f"= {B*1000*1.0737:7.0f} GB/s  -> {B*1000/HW_GIBS*100:5.0f}% of hardware peak"
          f"  {'PLAUSIBLE' if B*1000 <= HW_GIBS*1.1 else '*** IMPOSSIBLE ***'}")
    mbu1 = W / t1 * 1000 * 1.0737 / 800 * 100
    mbu2 = (W / 2) / t2 * 1000 * 1.0737 / 800 * 100
    print(f"{'':>10}  MBU: TP1 {mbu1:.0f}%   TP2 {mbu2:.0f}% per card\n")
