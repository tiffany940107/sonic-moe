# Copyright (c) 2026, SonicMoE contributors.
"""Four-GPU correctness gates for exact SM100 MXFP8 expert parallelism."""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
from sonicmoe import ExpertParallelMoE, KernelBackendMoE, MoE, Mxfp8SGD
from sonicmoe.enums import ActivationType
from sonicmoe.functional.ep4 import (
    dispatch_mxfp8,
    make_expert_parallel_dispatch_plan,
)

_EXPECTED_WORLD = 4
pytestmark = pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) != _EXPECTED_WORLD,
    reason="launch with torchrun --nproc-per-node=4",
)


@pytest.fixture(scope="session", autouse=True)
def _distributed_group():
    if int(os.environ.get("WORLD_SIZE", "1")) != _EXPECTED_WORLD:
        yield
        return
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    if torch.cuda.get_device_properties(local_rank).major != 10:
        pytest.skip("EP4 MXFP8 tests require SM100 GPUs")
    yield
    # No teardown barrier: synchronized assertions above already give peers a
    # coherent failure, while an extra barrier can mask the original traceback.
    dist.destroy_process_group()


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    numerator = (actual.float() - expected.float()).norm()
    denominator = expected.float().norm().clamp_min(1e-7)
    return (numerator / denominator).item()


def _assert_all_ranks(condition: bool, message: str, device: torch.device) -> None:
    """Fail a distributed test coherently instead of stranding peers in NCCL."""
    local = torch.tensor([condition], dtype=torch.uint8, device=device)
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    failing = [rank for rank, value in enumerate(gathered) if not bool(value.item())]
    assert not failing, f"{message}; failing ranks: {failing}"


def _full_model(
    *,
    bias: bool,
    training: bool = True,
    top_k: int = 2,
    hidden_size: int = 128,
    intermediate_size: int = 128,
) -> MoE:
    torch.manual_seed(1701)
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    model = MoE(
        num_experts=8,
        num_experts_per_tok=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation_function=ActivationType.SWIGLU,
        add_bias=bias,
        std=0.02,
    ).to(device=device, dtype=torch.bfloat16)
    if bias:
        torch.nn.init.normal_(model.c_fc.bias, std=0.01)
        torch.nn.init.normal_(model.c_proj.bias, std=0.01)
    model.train(training)
    return model


def _rank_input(
    tokens: int, *, requires_grad: bool = False, hidden_size: int = 128
) -> torch.Tensor:
    rank = dist.get_rank()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    generator = torch.Generator(device=device).manual_seed(1900 + rank)
    return torch.randn(
        tokens,
        hidden_size,
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
        requires_grad=requires_grad,
    )


def test_ep4_inference_matches_full_moe():
    full = _full_model(bias=False, training=False)
    ep = ExpertParallelMoE.from_moe(
        full,
        use_fused_route_pack=True,
        use_fused_reduce=True,
        use_comm_overlap=True,
    ).eval()
    x = _rank_input(37)
    with torch.inference_mode():
        expected = full(
            x,
            kernel_backend_moe=KernelBackendMoE.sonicmoe_mxfp8,
            is_inference_mode=True,
        )[0]
        actual = ep(x, is_inference_mode=True)[0]
    assert _relative_l2(actual, expected) < 0.10


def test_ep4_forward_backward_and_all_gradients():
    full = _full_model(bias=True)
    ep = ExpertParallelMoE.from_moe(
        full,
        use_fused_route_pack=True,
        use_fused_reduce=True,
        use_comm_overlap=True,
    )
    x_ref = _rank_input(31, requires_grad=True)
    x_ep = x_ref.detach().clone().requires_grad_()
    generator = torch.Generator(device=x_ref.device).manual_seed(2100 + dist.get_rank())
    grad = torch.randn(
        x_ref.shape,
        generator=generator,
        dtype=x_ref.dtype,
        device=x_ref.device,
    )

    expected, aux_ref = full(x_ref, kernel_backend_moe=KernelBackendMoE.sonicmoe_mxfp8)
    actual, aux_ep = ep(x_ep)
    assert _relative_l2(actual, expected) < 0.10
    loss_ref = (expected * grad).float().sum() + 0.01 * aux_ref.float()
    loss_ep = (actual * grad).float().sum() + 0.01 * aux_ep.float()
    loss_ref.backward()
    loss_ep.backward()

    assert _relative_l2(x_ep.grad, x_ref.grad) < 0.18
    assert _relative_l2(ep.router.weight.grad, full.router.weight.grad) < 0.18

    logical = ep.local_expert_ids.to(x_ref.device)
    for local_parameter, full_parameter in (
        (ep.c_fc.weight, full.c_fc.weight),
        (ep.c_fc.bias, full.c_fc.bias),
        (ep.c_proj.weight, full.c_proj.weight),
        (ep.c_proj.bias, full.c_proj.bias),
    ):
        dist.all_reduce(full_parameter.grad)
        expected_grad = full_parameter.grad.index_select(0, logical)
        assert torch.isfinite(local_parameter.grad).all()
        assert _relative_l2(local_parameter.grad, expected_grad) < 0.20


def test_ep4_bf16_reference_forward_backward():
    full = _full_model(bias=False)
    ep = ExpertParallelMoE.from_moe(
        full,
        expert_backend=KernelBackendMoE.sonicmoe,
    )
    x_ref = _rank_input(23, requires_grad=True)
    x_ep = x_ref.detach().clone().requires_grad_()
    generator = torch.Generator(device=x_ref.device).manual_seed(2300 + dist.get_rank())
    grad = torch.randn(
        x_ref.shape,
        generator=generator,
        dtype=x_ref.dtype,
        device=x_ref.device,
    )

    expected, aux_ref = full(x_ref, kernel_backend_moe=KernelBackendMoE.sonicmoe)
    actual, aux_ep = ep(x_ep)
    assert _relative_l2(actual, expected) < 0.01
    ((expected * grad).float().sum() + 0.01 * aux_ref.float()).backward()
    ((actual * grad).float().sum() + 0.01 * aux_ep.float()).backward()
    assert _relative_l2(x_ep.grad, x_ref.grad) < 0.02
    assert _relative_l2(ep.router.weight.grad, full.router.weight.grad) < 0.02

    logical = ep.local_expert_ids.to(x_ref.device)
    for local_parameter, full_parameter in (
        (ep.c_fc.weight, full.c_fc.weight),
        (ep.c_proj.weight, full.c_proj.weight),
    ):
        dist.all_reduce(full_parameter.grad)
        expected_grad = full_parameter.grad.index_select(0, logical)
        assert _relative_l2(local_parameter.grad, expected_grad) < 0.02


def test_ep4_empty_destination_ranks_and_optimizer_steps():
    full = _full_model(bias=True)
    with torch.no_grad():
        full.router.weight.zero_()
        full.router.weight[0, 0] = 2.0
        full.router.weight[1, 0] = 1.0
    ep = ExpertParallelMoE.from_moe(
        full,
        use_fused_route_pack=True,
        use_fused_reduce=True,
    )
    optimizer = Mxfp8SGD(ep, lr=0.1)
    initial = ep.c_fc.weight.detach().clone()
    ep.eval()
    with torch.inference_mode():
        inference_output = ep(
            torch.ones(5, 128, dtype=torch.bfloat16, device=initial.device),
            is_inference_mode=True,
        )[0]
    _assert_all_ranks(
        bool(torch.isfinite(inference_output).all().item()),
        "empty-destination inference produced a non-finite output",
        initial.device,
    )
    ep.train()
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        x = torch.zeros(17, 128, dtype=torch.bfloat16, device=initial.device)
        x[:, 0] = 1.0 + 0.1 * step
        output, aux_loss = ep(x)
        loss = output.float().sum() + 0.01 * aux_loss.float()
        _assert_all_ranks(
            bool(torch.isfinite(loss).item()),
            f"step {step} produced a non-finite loss",
            initial.device,
        )
        loss.backward()
        gradients_ok = all(
            parameter.grad is not None
            and bool(torch.isfinite(parameter.grad).all().item())
            for parameter in ep.parameters()
        )
        _assert_all_ranks(
            gradients_ok,
            f"step {step} produced a missing or non-finite gradient",
            initial.device,
        )
        ep.all_reduce_replicated_gradients_(average=True)
        optimizer.step()
    delta = (ep.c_fc.weight.float() - initial.float()).abs().max()
    rank_zero_delta = delta if dist.get_rank() == 0 else torch.zeros_like(delta)
    dist.broadcast(rank_zero_delta, src=0)
    _assert_all_ranks(
        rank_zero_delta.item() > 0,
        "rank 0 expert weights did not update",
        initial.device,
    )


def test_replicated_router_gradient_sync():
    full = _full_model(bias=False)
    ep = ExpertParallelMoE.from_moe(full)
    x = _rank_input(13)
    output, aux_loss = ep(x)
    (output.float().square().mean() + aux_loss.float()).backward()
    expected = ep.router.weight.grad.detach().clone()
    dist.all_reduce(expected)
    ep.all_reduce_replicated_gradients_()
    torch.testing.assert_close(ep.router.weight.grad, expected, rtol=0, atol=0)


def test_ep4_auto_route_pack_crossover():
    full = _full_model(bias=False, training=False)
    ep = ExpertParallelMoE.from_moe(
        full,
        route_pack_min_elements=2_000,
    ).eval()
    with torch.inference_mode():
        ep(_rank_input(13), is_inference_mode=True)
        small_enabled = ep._last_route_pack_enabled
        ep(_rank_input(17), is_inference_mode=True)
        large_enabled = ep._last_route_pack_enabled
    _assert_all_ranks(
        not small_enabled and large_enabled,
        "auto route-pack crossover selected the wrong path",
        full.c_fc.weight.device,
    )


def test_ep4_collective_placement_migration_preserves_output():
    full = _full_model(bias=True, training=False)
    ep = ExpertParallelMoE.from_moe(
        full,
        use_fused_route_pack=True,
        use_fused_reduce=True,
    ).eval()
    x = _rank_input(19)
    placement = (
        torch.arange(full.num_experts, dtype=torch.int64) % dist.get_world_size()
    )
    ep.set_expert_placement_(placement)
    with torch.inference_mode():
        expected = full(
            x,
            kernel_backend_moe=KernelBackendMoE.sonicmoe_mxfp8,
            is_inference_mode=True,
        )[0]
        actual = ep(x, is_inference_mode=True)[0]
    _assert_all_ranks(
        ep.placement_version.item() == 1 and _relative_l2(actual, expected) < 0.10,
        "collective expert placement migration changed the output",
        x.device,
    )


def test_ep4_topk_variants_masked_slots_and_remote_only_transport():
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    for top_k, hidden_size in ((1, 256), (4, 128)):
        full = _full_model(
            bias=False,
            training=False,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=hidden_size,
        )
        ep = ExpertParallelMoE.from_moe(
            full,
            use_fused_route_pack=True,
            use_fused_reduce=True,
            use_zero_material_qdata=True,
        ).eval()
        x = _rank_input(11, hidden_size=hidden_size)
        with torch.inference_mode():
            expected = full(
                x,
                kernel_backend_moe=KernelBackendMoE.sonicmoe_mxfp8,
                is_inference_mode=True,
            )[0]
            actual = ep(x, is_inference_mode=True)[0]
        _assert_all_ranks(
            _relative_l2(actual, expected) < 0.10,
            f"EP4 Top-{top_k} inference diverged from EP1",
            device,
        )

    rank = dist.get_rank()
    world = dist.get_world_size()
    destination = (rank + 1) % world
    expert_to_rank = (torch.arange(8, device=device) // 2).to(torch.int64)
    expert_to_local = (torch.arange(8, device=device) % 2).to(torch.int64)
    remote_start = destination * 2
    expert_ids = torch.tensor(
        [
            [remote_start, -1, -1, -1],
            [remote_start + 1, remote_start, -1, -1],
        ],
        dtype=torch.int32,
        device=device,
    )
    weights = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.6, 0.4, 0.0, 0.0]],
        dtype=torch.float32,
        device=device,
    )
    plan = make_expert_parallel_dispatch_plan(
        expert_ids,
        weights,
        expert_to_rank,
        expert_to_local,
    )
    expected_counts = tuple(2 if peer == destination else 0 for peer in range(world))
    plan_ok = (
        plan.send_counts == expected_counts
        and plan.total_send_tokens == 2
        and bool((plan.send_expert_ids[:, 2:] == -1).all().item())
        and bool((plan.send_weights[:, 2:] == 0).all().item())
    )
    _assert_all_ranks(plan_ok, "masked remote-only dispatch plan is invalid", device)

    x = _rank_input(2, requires_grad=True)
    dispatched = dispatch_mxfp8(x, plan, packed_transport=True)
    transport_ok = (
        dispatched.x.shape == (2, 128)
        and dispatched.expert_ids.shape == (2, 4)
        and bool((dispatched.expert_ids[:, 2:] == -1).all().item())
        and bool(torch.isfinite(dispatched.weights).all().item())
    )
    _assert_all_ranks(
        transport_ok,
        "masked remote-only MXFP8 transport corrupted its records",
        device,
    )
