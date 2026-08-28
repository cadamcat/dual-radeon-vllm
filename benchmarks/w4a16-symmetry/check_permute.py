"""Can the layout tool vLLM already has do the zp transpose, without repacking?

`permute_param_layout_` is the real function under test. Constructing a real
PackedvLLMParameter outside an engine needs a TP group and a vLLM config
context, so the parameter is stood in for by an object carrying the exact
attributes compressed_tensors_wNa16.py registers for the group-quantized
asymmetric case:

    qzeros = PackedvLLMParameter(input_dim=1, output_dim=0, packed_dim=0,
                                 packed_factor=self.pack_factor, ...)

and the exact shape the AWQ checkpoint ships, (N/8, groups) = (640, 544).
Pure CPU, no GPU, no model.
"""
import sys

import torch
from vllm.model_executor.parameter import permute_param_layout_

N, GROUPS, PACK = 5120, 544, 8


class ZP:                      # stands in for PackedvLLMParameter
    def __init__(self):
        self.data = torch.zeros((N // PACK, GROUPS), dtype=torch.int32)
        self._input_dim, self._output_dim, self._packed_dim = 1, 0, 0
    input_dim = property(lambda s: s._input_dim)
    output_dim = property(lambda s: s._output_dim)
    packed_dim = property(lambda s: s._packed_dim)


ok = True
zp = ZP()
print(f"checkpoint ships   : {tuple(zp.data.shape)}  input_dim={zp.input_dim} "
      f"output_dim={zp.output_dim} packed_dim={zp.packed_dim}")
try:
    permute_param_layout_(zp, input_dim=0, output_dim=1, packed_dim=1)
    print(f"after permute      : {tuple(zp.data.shape)}  input_dim={zp.input_dim} "
          f"output_dim={zp.output_dim} packed_dim={zp.packed_dim}")
    ok = tuple(zp.data.shape) == (GROUPS, N // PACK)
    print(f"PERMUTE_OK={ok}  (kernel wants {(GROUPS, N // PACK)})")
except Exception as exc:
    ok = False
    print(f"PERMUTE_FAILED {type(exc).__name__}: {exc}")

# the same call without packed_dim, which is what transform_w_s already does
zp2 = ZP()
permute_param_layout_(zp2, input_dim=0, output_dim=1)
print(f"without packed_dim : {tuple(zp2.data.shape)} (same permutation)")
# the verdict has to reach the caller: this printed CHECK_DONE and exited 0
# whether the permutation worked, failed, or produced the wrong shape
print("CHECK_DONE" if ok else "CHECK_FAILED")
sys.exit(0 if ok else 1)
