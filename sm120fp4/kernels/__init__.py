"""The stage-2 W4A16 MoE kernels (CUDA sources inline, compiled at first use through torch.utils.cpp_extension.load_inline).
moe_layer.py is the layer; the other modules are the kernels and helpers it loads by path from this directory."""
