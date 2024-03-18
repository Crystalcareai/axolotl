"""
Patches to support multipack for mixtral
"""
import torch

def patch_mixtral_moe_forward_zero3() -> None:
    from transformers.models.mixtral.modeling_mixtral import (
        MixtralBLockSparseTop2MLP,
        MixtralSparseMoeBlock,
    )
    from axolotl.monkeypatch.moe.mlp import FusedExperts
    from axolotl.monkeypatch.moe.moe import SparseMoeBlock

    def mixtral_fused_experts_forward(self, x: torch.Tensor, routing_weights: torch.Tensor, selected_experts: torch.Tensor):
        fused_experts = FusedExperts(self)
        return fused_experts.forward(x, routing_weights, selected_experts)

    def mixtral_sparse_moe_block_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        sparse_moe_block = SparseMoeBlock(self)
        return sparse_moe_block.forward(hidden_states)

    MixtralBLockSparseTop2MLP.forward = mixtral_fused_experts_forward
    MixtralSparseMoeBlock.forward = mixtral_sparse_moe_block_forward
