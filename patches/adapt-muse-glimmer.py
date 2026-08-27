"""Adapt muse_glimmer.py to this vLLM's weight-loading convention.

Upstream expresses the qkv / gate_up fusion through
WeightsMapper.orig_to_new_stacked, a field added after this container's vLLM was
built. Backporting that field would mean editing model_executor/models/utils.py
and layers/linear.py, which every one of the 143 models here shares, so instead
the same mapping is expressed the way this version does it everywhere else: a
stacked_params_mapping loop in the model's own load_weights.

Ordering is preserved: upstream applies the substring mapper before the stacked
fold, and the mapper still runs first here.

What keeps `.self_attn.gate_proj` from being folded into `gate_up_proj` is NOT
that ordering, and an earlier version of this note said it was. The mapper
renames it to `.self_attn.output_gate_proj`, which still contains the substring
`gate_proj` and still matches the fold table. The actual defence is the
`if mapped not in params_dict: continue` check in the loop below: the folded
name `output_gate_up_proj` does not exist on the model, so the mapping is
rejected and the weight loads under its own name. Do not remove that check on
the assumption the rename is doing the work. The unloaded-parameter
`raise RuntimeError` at the end is the second line of defence, and would turn
any such mistake into a loud failure rather than a silently wrong weight.
"""
import ast
import re
import shutil

P = "/opt/python/lib/python3.14/site-packages/vllm/model_executor/models/muse_glimmer.py"
BAK = P + ".pre-adapt"

NEW_LOAD = '''    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Adapted downstream: this vLLM's WeightsMapper has no
        # orig_to_new_stacked, so the fusion it described is done here instead.
        from vllm.model_executor.model_loader.weight_utils import default_weight_loader

        from .utils import is_pp_missing_parameter

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded: set[str] = set()

        for name, loaded_weight in self.hf_to_vllm_mapper.apply(weights):
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                mapped = name.replace(weight_name, param_name)
                if mapped.endswith(".bias") and mapped not in params_dict:
                    continue
                if is_pp_missing_parameter(mapped, self):
                    continue
                if mapped not in params_dict:
                    continue
                param = params_dict[mapped]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded.add(mapped)
                break
            else:
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded.add(name)

        # AutoWeightsLoader validates coverage; this hand-rolled loop does not,
        # and a parameter that never receives a weight keeps its initialised
        # values without raising. Report it instead of failing silently.
        missing = [n for n in params_dict if n not in loaded]
        print(
            f"[muse-load] params={len(params_dict)} loaded={len(loaded)} "
            f"missing={len(missing)}",
            flush=True,
        )
        # Counting is not checking. 1600 + 312 = 1912 holds just as well if a
        # real parameter went unloaded and an expected scale happened to load, so
        # assert what the missing names are, not how many there are. The expected
        # remainder is the KV-cache quantisation scales this checkpoint does not
        # carry, six per layer.
        EXPECTED = (".k_scale", ".v_scale", ".q_scale", ".prob_scale",
                    ".k_zp", ".v_zp")
        unexpected = [n for n in missing if not n.endswith(EXPECTED)]
        if unexpected:
            raise RuntimeError(
                f"[muse-load] {len(unexpected)} parameter(s) received no weight and "
                f"are not KV-cache scales: {unexpected[:12]}. They keep their "
                f"initialised values, so this model would run and be wrong."
            )
        if missing:
            print(f"[muse-load] all {len(missing)} missing are KV-cache scales, "
                  f"e.g. {missing[:3]}", flush=True)
        return loaded
'''


def main():
    src = open(BAK).read() if __import__("os").path.exists(BAK) else open(P).read()
    if not __import__("os").path.exists(BAK):
        shutil.copyfile(P, BAK)

    # 1. drop the orig_to_new_stacked block from the mapper
    pat = re.compile(r"\n        orig_to_new_stacked=\{.*?\n        \},", re.S)
    src2, n = pat.subn("", src)
    assert n == 1, f"orig_to_new_stacked block matched {n} times"

    # 2. replace the AutoWeightsLoader delegation with the classic loop
    old = ("    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:\n"
           "        loader = AutoWeightsLoader(self)\n"
           "        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)\n")
    assert old in src2, "load_weights body not found verbatim"
    src3 = src2.replace(old, NEW_LOAD, 1)


    # 3. is_vit_use_data_parallel() takes no arguments in this version; upstream
    #    added a divisibility fallback. Reproduce it inline so behaviour matches.
    old_call = "is_vit_use_data_parallel(num_heads)"
    new_call = ("(is_vit_use_data_parallel()\n"
                "            or num_heads % get_tensor_model_parallel_world_size() != 0)")
    n3 = src3.count(old_call)
    assert n3 >= 1, "is_vit_use_data_parallel(num_heads) call not found"
    src3 = src3.replace(old_call, new_call)
    print(f"is_vit_use_data_parallel 调用点已改写 x{n3}")

    ast.parse(src3)
    open(P, "w").write(src3)
    print("orig_to_new_stacked 已移除，load_weights 已改写为经典循环")
    print("ast.parse 通过，备份在", BAK)


if __name__ == "__main__":
    main()
