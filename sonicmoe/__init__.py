# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

__version__ = "0.1.2.post1"

from .enums import KernelBackendMoE
from .functional import moe_general_routing_inputs, moe_TC_softmax_topk_layer
from .functional.mxfp8 import (
    Mxfp8MoEKernelConfig,
    Mxfp8MoEWorkspace,
    allocate_mxfp8_moe_workspace,
    allocate_mxfp8_weights,
    dequantize_varlen_m_operand,
    make_varlen_m_operand,
    moe_mxfp8_grouped_forward,
    mxfp8_down_grouped,
    mxfp8_down_grouped_weighted_reduce,
    mxfp8_grouped_gemm,
    mxfp8_swiglu_grouped,
    mxfp8_swiglu_grouped_out,
    pack_varlen_m_scales,
    quantize_mxfp8_rows,
    quantize_varlen_m_operand,
    unpack_varlen_m_scales,
)
from .functional.mxfp8_route_pack import (
    Mxfp8RoutePackResult,
    Mxfp8RoutePackWorkspace,
    allocate_mxfp8_route_pack_workspace,
    route_pack_mxfp8,
)
from .functional.mxfp8_weighted_reduce import (
    Mxfp8WeightedReduceWorkspace,
    allocate_mxfp8_weighted_reduce_workspace,
    hybrid_weighted_reduce_mxfp8,
    segmented_weighted_reduce_mxfp8,
    weighted_reduce_mxfp8,
)
from .moe import MoE

__all__ = [
    "KernelBackendMoE",
    "MoE",
    "Mxfp8MoEKernelConfig",
    "Mxfp8MoEWorkspace",
    "Mxfp8RoutePackResult",
    "Mxfp8RoutePackWorkspace",
    "allocate_mxfp8_moe_workspace",
    "allocate_mxfp8_route_pack_workspace",
    "allocate_mxfp8_weighted_reduce_workspace",
    "allocate_mxfp8_weights",
    "dequantize_varlen_m_operand",
    "make_varlen_m_operand",
    "moe_TC_softmax_topk_layer",
    "moe_general_routing_inputs",
    "moe_mxfp8_grouped_forward",
    "mxfp8_down_grouped",
    "mxfp8_down_grouped_weighted_reduce",
    "mxfp8_grouped_gemm",
    "mxfp8_swiglu_grouped",
    "mxfp8_swiglu_grouped_out",
    "pack_varlen_m_scales",
    "quantize_mxfp8_rows",
    "quantize_varlen_m_operand",
    "route_pack_mxfp8",
    "hybrid_weighted_reduce_mxfp8",
    "segmented_weighted_reduce_mxfp8",
    "weighted_reduce_mxfp8",
    "Mxfp8WeightedReduceWorkspace",
    "unpack_varlen_m_scales",
]
