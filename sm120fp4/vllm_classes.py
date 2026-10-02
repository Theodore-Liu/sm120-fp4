"""The vLLM config and method subclasses of the sm120fp4 backend, at module level.

They live in their own module, imported only when vLLM is present, because vLLM pickles the quantization config into
its engine-core subprocess: a class defined inside a function cannot be pickled ("Can't get local object"), which is how
the first model-level run failed. :mod:`sm120fp4.vllm_backend` holds the vLLM-free parts (layout rotation, the forward
over the kernels) and the ``register`` entry point; see its docstring for how the backend is selected.
"""
from __future__ import annotations

import torch
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config, ModelOptNvFp4FusedMoE

from sm120fp4.vllm_backend import HIDDEN, INTERMEDIATES, Weights, _log, kernels, weights_from_vllm_layout


class ModelOptNvFp4FusedMoESM120(ModelOptNvFp4FusedMoE):
    """vLLM's NVFP4 MoE method with the repository's decode layer in place of the Marlin repack and kernel.

    The class name keeps "ModelOpt" in it on purpose: RoutedExperts.weight_loader branches on
    ``self.quant_method.__class__.__name__`` to pick the ModelOpt loading path for input scales and weight scales
    (``"ModelOpt" in quant_method_name``); a method named otherwise falls into the generic scale branch, finds no
    ``quant_method`` attribute on the parameter and raises "quant method must be one of [...]" at load - the second
    model-level run failed that way."""

    def __init__(self, quant_config, moe_config) -> None:
        # Not the parent's __init__: it selects a vLLM backend (Marlin for W4A16) that would never run here.
        FusedMoEMethodBase.__init__(self, moe_config)
        self.quant_config = quant_config
        self.use_a16 = True
        self.use_global_sf = False  # input scales are per layer, as the parent allocates them without global sf
        self.nvfp4_backend, self.experts_cls = None, None
        self.weights: Weights | None = None

    @staticmethod
    def fits(moe_config) -> tuple[bool, str]:
        """Whether the layer described by a FusedMoEConfig can run on the kernels; (False, reason) otherwise."""
        if not getattr(moe_config, "is_act_and_mul", False) or moe_config.activation != MoEActivation.SILU:
            return False, f"activation {moe_config.activation!r} is not gated SiLU"
        if moe_config.hidden_dim != HIDDEN:
            return False, f"hidden size {moe_config.hidden_dim} (kernels take {HIDDEN})"
        if moe_config.intermediate_size_per_partition not in INTERMEDIATES:
            return False, f"intermediate size {moe_config.intermediate_size_per_partition} (kernels take {INTERMEDIATES})"
        if moe_config.moe_parallel_config.use_ep:
            return False, "expert parallelism (the layer has no expert map)"
        if moe_config.in_dtype != torch.bfloat16:
            return False, f"activations {moe_config.in_dtype} (kernels take bf16)"
        return True, ""

    @property
    def is_monolithic(self) -> bool:
        return False

    @property
    def supports_eplb(self) -> bool:
        return False

    def get_fused_moe_quant_config(self, layer):
        return None

    def process_weights_after_loading(self, layer) -> None:
        self.weights = weights_from_vllm_layout(layer.w13_weight.data, layer.w13_weight_scale.data,
                                                layer.w13_weight_scale_2.data, layer.w2_weight.data,
                                                layer.w2_weight_scale.data, layer.w2_weight_scale_2.data,
                                                int(layer.top_k))
        w = self.weights
        # Keep the parameters the engine may introspect, in the kernels' layout, as plain non-trainable parameters;
        # the activation scales of a W4A4 path are not used by W4A16 kernels.
        for name, t in (("w13_weight", w.q1), ("w13_weight_scale", w.s1.view(torch.float8_e4m3fn)),
                        ("w13_weight_scale_2", w.alpha1), ("w2_weight", w.q2),
                        ("w2_weight_scale", w.s2.view(torch.float8_e4m3fn)), ("w2_weight_scale_2", w.alpha2)):
            layer.register_parameter(name, torch.nn.Parameter(t, requires_grad=False))
        for name in ("w13_input_scale", "w2_input_scale"):
            if hasattr(layer, name):
                layer.register_parameter(name, None)
        kernels()  # compile now, not on the first request

    def apply(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
        assert self.weights is not None, "process_weights_after_loading has not run"
        if shared_experts is not None:
            raise NotImplementedError("shared experts are not run by this backend")
        if getattr(layer, "apply_router_weight_on_input", False):
            raise NotImplementedError("apply_router_weight_on_input")
        if getattr(layer, "expert_map", None) is not None:
            raise NotImplementedError("expert map (expert parallelism)")
        return self.weights.forward(x, topk_ids, topk_weights)


class SM120Fp4Config(ModelOptNvFp4Config):
    """vLLM's ModelOpt NVFP4 config, routing the routed-experts layers the kernels fit to the method above."""

    def get_quant_method(self, layer, prefix: str):
        if isinstance(layer, RoutedExperts) and not self.is_layer_excluded(prefix):
            ok, why = ModelOptNvFp4FusedMoESM120.fits(layer.moe_config)
            if ok:
                return ModelOptNvFp4FusedMoESM120(quant_config=self, moe_config=layer.moe_config)
            _log(f"layer {prefix} stays on vLLM's path: {why}")
        return super().get_quant_method(layer, prefix)


SM120Fp4MoEMethod = ModelOptNvFp4FusedMoESM120  # the short name the tests and docs use
