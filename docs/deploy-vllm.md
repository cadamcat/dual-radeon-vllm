# Deploying into a ROCm / vLLM container

Removing the hostcall requirement is necessary but **not sufficient**. Two
further traps sit between a correct `librccl.so` and a working vLLM, and both
present as something completely unrelated. This page exists so you do not lose a
day to either.

---

## The three pieces

| # | Piece | Failure if you skip it |
|---|---|---|
| 1 | No-hostcall `librccl.so` | `the operation cannot be performed in the present state` at the first collective |
| 2 | `rsmi` stub + `patchelf` | `torch.cuda.device_count()` returns **0** — looks like the GPUs vanished |
| 3 | `sitecustomize.py` | vLLM: *"Failed to infer device type"* — looks like a vLLM/platform bug |

---

## Trap 1 — `librocm_smi64` poisons `amdsmi` enumeration

**Symptom:** you swap in your rebuilt RCCL and now `torch.cuda.device_count()` is
`0`. Meanwhile a direct `hipGetDeviceCount()` still returns 2. The GPUs are fine.

**Cause:** ROCm-PyTorch enumerates devices through **`amdsmi`**. RCCL's `NEEDED`
entry on the older-style `librocm_smi64.so.1` pulls that library into the
process, and once it is loaded `amdsmi`'s enumeration returns zero. The poison
travels through global symbol visibility — loading the same library with
`RTLD_LOCAL` is harmless, which is how we localised it.

**Fix:**

```bash
patchelf --remove-needed librocm_smi64.so.1 librccl-final.so
```

RCCL has an `alt_rsmi` fallback path that reads sysfs, so it does not need the
real library.

**But** removing it leaves **9 undefined `rsmi_*` symbols**. Hence:

```bash
patchelf --add-needed librsmi_stub.so.1 librccl-final.so
```

`deploy/rsmi_stub.c` defines all 9 as no-ops returning an error code, which
makes RCCL take the `alt_rsmi` path deliberately. Put the stub in the **same
directory** as `librccl` so `$ORIGIN` resolves it.

> Any experiment that swaps `librccl` **must** do this first, or you will get a
> false failure and blame the wrong thing.

---

## Trap 2 — the stub shadows the real rsmi, breaking `amdsmi`

**Symptom:** device count is now correct, but vLLM aborts with
*"Failed to infer device type"*.

**Cause:** the stub's `rsmi_*` symbols are global, so they also shadow the real
`rsmi_*` that `amdsmi` (and `goamdsmi_shim`) call internally. `amdsmi` therefore
returns `AMDSMI_STATUS_NOT_INIT`, and vLLM's platform detection fails.

**Fix:** bind the real rsmi *before* torch loads RCCL and the stub. Python
imports `sitecustomize` automatically at startup, which is early enough:

```python
try:
    import amdsmi
    amdsmi.amdsmi_init()
except Exception:
    pass
```

⚠️ **Do not call `amdsmi_shut_down()`.** Keeping `amdsmi` initialised holds the
native libraries loaded, which keeps rsmi bound to the real implementation for
the life of the process. Shutting it down releases the binding and the stub wins
again.

---

## Putting it together

```bash
# inside a ROCm/vLLM container with this repo mounted
./deploy/deploy-tp2.sh
```

`deploy-tp2.sh` compiles the stub, swaps `librccl` into **both** ROCm SDK library
paths that torch may load from, installs `sitecustomize.py` into `site-packages`,
and verifies `device_count == 2`.

Then launch:

```bash
NCCL_P2P_DISABLE=1 HSA_ENABLE_SDMA=0 \
vllm serve <model> \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90
```

`NCCL_P2P_DISABLE=1` and `HSA_ENABLE_SDMA=0` are for our no-P2P cross-die
topology; they are **not** part of the hostcall fix. Drop them if your topology
supports P2P.

---

## Verifying it actually worked

```bash
# 1. the library carries no hostcall
./build/verify-nohostcall.sh /path/to/librccl.so.1

# 2. torch still sees both GPUs (trap 1)
python3 -c "import torch; print(torch.cuda.device_count())"   # 2

# 3. a real collective completes (30 s, no model loading)
torchrun --nproc_per_node=2 diagnose/ar.py
```

Expected from step 3:

```
[rank0] all_reduce OK sum=2.0
[rank1] all_reduce OK sum=2.0
```

Use `ar.py` rather than starting vLLM to test — it exercises the identical code
path in about 30 seconds instead of several minutes of weight loading.

---

## Notes

- Instruction-tuned models must go through `/v1/chat/completions` so the chat
  template is applied. A bare `/v1/completions` call can produce garbage output —
  that is not a symptom of this bug.
- Hybrid Mamba/SSM models allocate one cache block per sequence; if
  `max_num_seqs` exceeds the available blocks, graph capture fails. Unrelated to
  hostcall, but easy to confuse with it.
