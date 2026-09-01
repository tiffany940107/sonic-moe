from __future__ import annotations

import torch

from ep4_support.transport import (
    DispatchPlan,
    allocate_transport_workspace,
    combine_deduplicated_out,
    dispatch_deduplicated_out,
)


def _world_one_plan() -> DispatchPlan:
    token_index = torch.tensor([2, 0, 3], dtype=torch.int64)
    return DispatchPlan(
        send_counts=[3],
        recv_counts=[3],
        send_token_indices=[token_index],
        send_expert_ids=torch.tensor([[0, -1], [1, 0], [0, 1]], dtype=torch.int32),
        send_weights=torch.tensor(
            [[0.75, 0.0], [0.6, 0.4], [0.25, 0.75]], dtype=torch.float32
        ),
        local_expert_map=torch.tensor([0, 1], dtype=torch.int64),
        send_token_indices_flat=token_index,
    )


def test_static_transport_workspace_reuses_storage(monkeypatch) -> None:
    def copy_all_to_all(out, tensor, **_kwargs):
        out.copy_(tensor)

    monkeypatch.setattr(torch.distributed, "all_to_all_single", copy_all_to_all)
    plan = _world_one_plan()
    x = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    scales = torch.arange(12, dtype=torch.uint8).reshape(4, 3)
    workspace = allocate_transport_workspace(
        plan, x, scales, source_tokens=4, output_hidden=2, output_dtype=torch.float32
    )
    pointers = tuple(
        tensor.data_ptr()
        for tensor in (
            workspace.send_x,
            workspace.recv_x,
            workspace.recv_ids,
            workspace.recv_weights,
            workspace.send_scales,
            workspace.recv_scales,
            workspace.returned,
            workspace.output,
        )
    )

    dispatched = dispatch_deduplicated_out(x, plan, workspace, scales)
    torch.testing.assert_close(dispatched.x, x[plan.send_token_indices_flat])
    torch.testing.assert_close(dispatched.expert_ids, plan.send_expert_ids)
    torch.testing.assert_close(dispatched.weights, plan.send_weights)
    torch.testing.assert_close(dispatched.scales, scales[plan.send_token_indices_flat])

    reduced = torch.tensor([[1.0, 2.0], [4.0, 8.0], [16.0, 32.0]])
    combined = combine_deduplicated_out(reduced, plan, 4, workspace)
    expected = torch.zeros((4, 2))
    expected.index_add_(0, plan.send_token_indices_flat, reduced)
    torch.testing.assert_close(combined, expected)

    dispatch_deduplicated_out(x + 1, plan, workspace, scales + 1)
    combine_deduplicated_out(reduced + 1, plan, 4, workspace)
    assert pointers == tuple(
        tensor.data_ptr()
        for tensor in (
            workspace.send_x,
            workspace.recv_x,
            workspace.recv_ids,
            workspace.recv_weights,
            workspace.send_scales,
            workspace.recv_scales,
            workspace.returned,
            workspace.output,
        )
    )
