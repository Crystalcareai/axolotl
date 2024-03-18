# axolotl_mixtral_moe_patch.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.models.mixtral.configuration_mixtral import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock, MixtralBlockSparseTop2MLP
from axolotl.monkeypatch.moe.ops import flatten_and_sort, padded_block_indices, scatter2scatter

class AxolotlMixtralSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        
        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        
        self.w1 = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.ffn_dim))
        self.w2 = nn.Parameter(torch.empty(self.num_experts, self.ffn_dim, self.hidden_dim))
        self.w3 = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.ffn_dim))
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        top_k_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        top_k_weights /= top_k_weights.sum(dim=-1, keepdim=True)
        top_k_weights = top_k_weights.to(hidden_states.dtype)
        
        with torch.no_grad():
            (sorted_expert_idxs, sorted_scattered_idxs) = flatten_and_sort(selected_experts)
            (padded_block_idxs, expert_offsets) = padded_block_indices(sorted_expert_idxs, self.num_experts)
        
        h = scatter2scatter(
            X=hidden_states,
            W=self.w1,
            sorted_expert_idxs=sorted_expert_idxs,
            sorted_scattered_idxs=sorted_scattered_idxs,
            padded_block_idxs=padded_block_idxs,
            k=self.top_k,
        )
        h = F.silu(h) * scatter2scatter(
            X=hidden_states,
            W=self.w3,
            sorted_expert_idxs=sorted_expert_idxs,
            sorted_scattered_idxs=sorted_scattered_idxs,
            padded_block_idxs=padded_block_idxs,
            k=self.top_k,
        )
        y = scatter2scatter(
            X=h,
            W=self.w2,
            sorted_expert_idxs=sorted_expert_idxs,
            sorted_scattered_idxs=sorted_scattered_idxs,
            padded_block_idxs=padded_block_idxs,
            k=1,
            gates=top_k_weights,
        )
        
        final_hidden_states = y.view(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

def replace_moe_block(model):
    modules_to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, MixtralSparseMoeBlock):
            config = MixtralConfig(
                hidden_size=module.hidden_dim,
                intermediate_size=module.ffn_dim,
                num_local_experts=module.num_experts,
                num_experts_per_tok=module.top_k,
                hidden_act="silu",
            )
            axolotl_moe_block = AxolotlMixtralSparseMoeBlock(config)
            modules_to_replace.append((name, axolotl_moe_block))
    
    for name, axolotl_moe_block in modules_to_replace:
        setattr(model, name, axolotl_moe_block)

def patch_model(model):
    replace_moe_block(model)
    
    modules_to_remove = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Module) and "MixtralBlockSparseTop2MLP" in str(type(module)):
            modules_to_remove.append(name)
    
    for name in modules_to_remove:
        setattr(model, name, None)
