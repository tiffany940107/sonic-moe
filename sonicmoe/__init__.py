# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

__version__ = "0.1.2.post1"

from .enums import KernelBackendMoE
from .functional import moe_general_routing_inputs, moe_TC_softmax_topk_layer
from .functional.mxfp8 import (
    Mxfp8MoEKernelConfig,
    allocate_mxfp8_weights,
    dequantize_varlen_m_operand,
    make_varlen_m_operand,
    moe_mxfp8_grouped_forward,
    mxfp8_down_grouped,
    mxfp8_grouped_gemm,
    mxfp8_swiglu_grouped,
    pack_varlen_m_scales,
    quantize_mxfp8_rows,
    quantize_varlen_m_operand,
    unpack_varlen_m_scales,
)
from .moe import MoE

__all__ = [
    "KernelBackendMoE",
    "MoE",
    "Mxfp8MoEKernelConfig",
    "allocate_mxfp8_weights",
    "dequantize_varlen_m_operand",
    "make_varlen_m_operand",
    "moe_TC_softmax_topk_layer",
    "moe_general_routing_inputs",
    "moe_mxfp8_grouped_forward",
    "mxfp8_down_grouped",
    "mxfp8_grouped_gemm",
    "mxfp8_swiglu_grouped",
    "pack_varlen_m_scales",
    "quantize_mxfp8_rows",
    "quantize_varlen_m_operand",
    "unpack_varlen_m_scales",
]
