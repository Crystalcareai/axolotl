"""
Adapted from:
https://github.com/shawntan/scattermoe
https://arxiv.org/abs/2403.08245
"""

import torch
from torch import nn

from axolotl.monkeypatch.moe import ops
from axolotl.monkeypatch.moe.linear import ParallelExperts


class FusedExperts(nn.Module):
    def __init__(
        self,
        experts=None,
        hidden_dim=128,
        ffn_dim=512,
        num_experts=8,
        top_k=2,
        activation=nn.SiLU(),
    ):
        """
        This implements fused experts that are compatible with Mixtral.
        MLP of type Gated-Linear Unit, typically with a SiLU activation function.
        """
        super(FusedExperts, self).__init__()
        expert_device = experts[0].w1.weight.device
        output_expert_device = experts[0].w2.weight.device

        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.experts = ParallelExperts(num_experts, hidden_dim, 2 * ffn_dim, expert_device)
        self.output_experts = ParallelExperts(num_experts, ffn_dim, hidden_dim, output_expert_device)
        self.top_k = min(top_k, self.num_experts)
        self.activation = activation

        # parallelize all w1 and w3 computation by concat + stack
        with torch.no_grad():
            torch.stack(
                [
                    torch.cat([experts[i].w1.weight, experts[i].w3.weight], dim=0)
                    for i in range(len(experts))
                ],
                dim=0,
                out=self.experts.weight.data,
            )

            # parallelize all w2 computation by stack
            torch.stack(
                [expert.w2.weight for expert in experts],
                dim=0,
                out=self.output_experts.weight.data,
            )

    def forward(
        self, x: torch.Tensor, routing_weights: torch.Tensor, selected_experts: torch.Tensor
    ):
        x_shape = x.size()
        x = x.view(-1, x_shape[-1])
        with torch.no_grad():
            sorted_expert_idxs, sorted_scattered_idxs = ops.flatten_and_sort(
                selected_experts
            )
            padded_block_idxs, expert_offsets = ops.padded_block_indices(
                sorted_expert_idxs, self.num_experts
            )

        h, gates = self.experts(
            x,
            self.top_k,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            padded_block_idxs,
            expert_offsets,
            grouped_out=True,
        ).chunk(2, dim=-1)
        h = self.activation(gates) * h
        y = self.output_experts(
            h,
            1,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            padded_block_idxs,
            expert_offsets,
            grouped_in=True,
            gates=routing_weights,
        )
        y = y.view(*x_shape[:-1], y.size(-1))
        return y