# Copyright (c) 2026, SonicMoE contributors.
"""End-to-end SM100 MXFP8 MoE training tests."""

import copy

import pytest
import torch
from sonicmoe import KernelBackendMoE, MoE
from sonicmoe.enums import ActivationType
from sonicmoe.functional.mxfp8 import Mxfp8Workspace


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    numerator = (actual.float() - expected.float()).norm()
    denominator = expected.float().norm().clamp_min(1e-7)
    return (numerator / denominator).item()


@pytest.mark.parametrize("add_bias", [False, True])
def test_sm100_mxfp8_moe_forward_backward(add_bias):
    if torch.cuda.get_device_properties(0).major != 10:
        pytest.skip("the single-GPU MXFP8 training backend targets SM100")
    torch.manual_seed(42)
    model_ref = (
        MoE(
            num_experts=8,
            num_experts_per_tok=2,
            hidden_size=256,
            intermediate_size=256,
            activation_function=ActivationType.SWIGLU,
            add_bias=add_bias,
            std=0.02,
        )
        .cuda()
        .to(torch.bfloat16)
    )
    if add_bias:
        torch.nn.init.normal_(model_ref.c_fc.bias, std=0.01)
        torch.nn.init.normal_(model_ref.c_proj.bias, std=0.01)
    model_mx = copy.deepcopy(model_ref)

    x_ref = (
        torch.randn(256, 256, device="cuda", dtype=torch.bfloat16) * 0.02
    ).requires_grad_()
    x_mx = x_ref.detach().clone().requires_grad_()
    grad_out = torch.randn_like(x_ref) * 0.02

    with torch.autocast("cuda", torch.float32):
        y_ref = model_ref(x_ref, kernel_backend_moe=KernelBackendMoE.torch)[0]
        y_mx = model_mx(x_mx, kernel_backend_moe=KernelBackendMoE.sonicmoe_mxfp8)[0]
    assert _relative_l2(y_mx, y_ref) < 0.08

    params_ref = list(model_ref.parameters())
    params_mx = list(model_mx.parameters())
    grads_ref = torch.autograd.grad(y_ref, [x_ref, *params_ref], grad_out)
    grads_mx = torch.autograd.grad(y_mx, [x_mx, *params_mx], grad_out)
    labels = ["input", *[name for name, _ in model_ref.named_parameters()]]
    for label, actual, expected in zip(labels, grads_mx, grads_ref):
        assert torch.isfinite(actual).all(), label
        assert _relative_l2(actual, expected) < 0.15, label


def test_sm100_mxfp8_empty_experts_and_non_aligned_m():
    """A tiny token count forces repeated offsets for most experts and gives
    active experts non-32-aligned M segments."""
    if torch.cuda.get_device_properties(0).major != 10:
        pytest.skip("the single-GPU MXFP8 training backend targets SM100")
    torch.manual_seed(7)
    model_ref = (
        MoE(
            num_experts=16,
            num_experts_per_tok=2,
            hidden_size=128,
            intermediate_size=128,
            activation_function=ActivationType.SWIGLU,
            add_bias=False,
            std=0.02,
        )
        .cuda()
        .to(torch.bfloat16)
    )
    with torch.no_grad():
        model_ref.router.weight.zero_()
    model_mx = copy.deepcopy(model_ref)
    x_ref = torch.randn(5, 128, device="cuda", dtype=torch.bfloat16).requires_grad_()
    x_mx = x_ref.detach().clone().requires_grad_()
    grad_out = torch.randn_like(x_ref)
    with torch.autocast("cuda", torch.float32):
        y_ref = model_ref(x_ref, kernel_backend_moe=KernelBackendMoE.torch)[0]
        y_mx = model_mx(x_mx, kernel_backend_moe=KernelBackendMoE.sonicmoe_mxfp8)[0]
    assert _relative_l2(y_mx, y_ref) < 0.08
    grads_ref = torch.autograd.grad(y_ref, [x_ref, *model_ref.parameters()], grad_out)
    grads_mx = torch.autograd.grad(y_mx, [x_mx, *model_mx.parameters()], grad_out)
    for actual, expected in zip(grads_mx, grads_ref):
        assert torch.isfinite(actual).all()
        assert _relative_l2(actual, expected) < 0.15


def test_sm100_mxfp8_optimizer_steps():
    if torch.cuda.get_device_properties(0).major != 10:
        pytest.skip("the single-GPU MXFP8 training backend targets SM100")
    torch.manual_seed(11)
    model = (
        MoE(
            num_experts=4,
            num_experts_per_tok=2,
            hidden_size=128,
            intermediate_size=128,
            activation_function=ActivationType.SWIGLU,
            add_bias=True,
            std=0.02,
        )
        .cuda()
        .to(torch.bfloat16)
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    initial = model.c_fc.weight.detach().clone()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
        output, aux_loss = model(x, kernel_backend_moe=KernelBackendMoE.sonicmoe_mxfp8)
        loss = output.float().square().mean() + 0.01 * aux_loss.float()
        assert torch.isfinite(loss)
        loss.backward()
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.parameters()
        )
        optimizer.step()
    assert not torch.equal(model.c_fc.weight, initial)


@pytest.mark.parametrize(
    "activation", [ActivationType.SWIGLU, ActivationType.GEGLU, ActivationType.REGLU]
)
@pytest.mark.parametrize("top_k", [1, 4])
def test_sm100_mxfp8_activation_and_topk_coverage(activation, top_k):
    if torch.cuda.get_device_properties(0).major != 10:
        pytest.skip("the single-GPU MXFP8 training backend targets SM100")
    torch.manual_seed(19 + top_k)
    model_ref = (
        MoE(
            num_experts=8,
            num_experts_per_tok=top_k,
            hidden_size=128,
            intermediate_size=128,
            activation_function=activation,
            add_bias=False,
            std=0.02,
        )
        .cuda()
        .to(torch.bfloat16)
    )
    model_mx = copy.deepcopy(model_ref)
    x_ref = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16).requires_grad_()
    x_mx = x_ref.detach().clone().requires_grad_()
    grad_out = torch.randn_like(x_ref)
    with torch.autocast("cuda", torch.float32):
        y_ref = model_ref(x_ref, kernel_backend_moe=KernelBackendMoE.torch)[0]
        y_mx = model_mx(x_mx, kernel_backend_moe=KernelBackendMoE.sonicmoe_mxfp8)[0]
    assert _relative_l2(y_mx, y_ref) < 0.08
    grads_ref = torch.autograd.grad(y_ref, [x_ref, *model_ref.parameters()], grad_out)
    grads_mx = torch.autograd.grad(y_mx, [x_mx, *model_mx.parameters()], grad_out)
    for actual, expected in zip(grads_mx, grads_ref):
        assert torch.isfinite(actual).all()
        assert _relative_l2(actual, expected) < 0.16


def test_sm100_mxfp8_inference_with_bias_matches_reference():
    if torch.cuda.get_device_properties(0).major != 10:
        pytest.skip("the single-GPU MXFP8 inference backend targets SM100")
    torch.manual_seed(29)
    model_ref = (
        MoE(
            num_experts=8,
            num_experts_per_tok=2,
            hidden_size=256,
            intermediate_size=256,
            activation_function=ActivationType.SWIGLU,
            add_bias=True,
            std=0.02,
        )
        .cuda()
        .to(torch.bfloat16)
        .eval()
    )
    torch.nn.init.normal_(model_ref.c_fc.bias, std=0.01)
    torch.nn.init.normal_(model_ref.c_proj.bias, std=0.01)
    model_mx = copy.deepcopy(model_ref)
    x = torch.randn(127, 256, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode(), torch.autocast("cuda", torch.float32):
        expected = model_ref(x, kernel_backend_moe=KernelBackendMoE.torch)[0]
        actual = model_mx(
            x,
            kernel_backend_moe=KernelBackendMoE.sonicmoe_mxfp8,
            is_inference_mode=True,
        )[0]
    assert _relative_l2(actual, expected) < 0.08


def test_sm100_mxfp8_workspace_reuses_and_invalidates_weight_buffers():
    if torch.cuda.get_device_properties(0).major != 10:
        pytest.skip("the single-GPU MXFP8 training backend targets SM100")
    torch.manual_seed(31)
    master = torch.randn(4, 256, 128, dtype=torch.bfloat16, device="cuda")
    weight = master.permute(1, 2, 0)
    workspace = Mxfp8Workspace()
    first = workspace.quantize_weight("test", weight)
    before = first.qdata.clone()
    second = workspace.quantize_weight("test", weight)
    assert second is first
    qdata_ptr, scale_ptr = second.qdata.data_ptr(), second.scale.data_ptr()
    with torch.no_grad():
        master.add_(0.25)
    updated = workspace.quantize_weight("test", weight)
    assert updated.qdata.data_ptr() == qdata_ptr
    assert updated.scale.data_ptr() == scale_ptr
    assert not torch.equal(updated.qdata, before)
