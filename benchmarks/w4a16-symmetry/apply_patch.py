"""Apply the three-line fix to the installed vllm, in place."""
import ast, pathlib
import vllm.model_executor.kernels.linear.mixed_precision.rdna3_w4a16 as RK

p = pathlib.Path(RK.__file__)
src = p.read_text()
old1 = "    SUPPORTED_QUANT_TYPES = [scalar_types.uint4b8]\n"
assert src.count(old1) == 1
src = src.replace(old1, "    SUPPORTED_QUANT_TYPES = [scalar_types.uint4b8, scalar_types.uint4]\n")
anchor = "        # Act-order: convert g_idx to the inverse permutation array exllama\n"
assert src.count(anchor) == 1
src = src.replace(anchor,
    "        if c.zero_points:\n\n"
    "            def transform_w_zp(x):\n"
    "                assert isinstance(x, BasevLLMParameter)\n"
    "                permute_param_layout_(x, input_dim=0, output_dim=1,\n"
    "                                      packed_dim=1)\n"
    "                x.data = x.data.contiguous()\n"
    "                return x\n\n"
    "            self._transform_param(layer, self.w_zp_name, transform_w_zp)\n\n" + anchor)
old3 = "        output = ops.gptq_gemm_rdna3(x_2d, w_q, w_zp, w_s, w_g_idx, False)\n"
assert src.count(old3) == 1
src = src.replace(old3,
    "        # GPTQ stores zeros under the v1 '+1' convention (uint4b8 carries\n"
    "        # the bias); compressed-tensors asymmetric checkpoints store the\n"
    "        # true zero (uint4, no bias) and want v2.\n"
    "        output = ops.gptq_gemm_rdna3(x_2d, w_q, w_zp, w_s, w_g_idx,\n"
    "                                     not c.weight_type.has_bias())\n")
p.write_text(src); ast.parse(src)
print("PATCH_APPLIED")
