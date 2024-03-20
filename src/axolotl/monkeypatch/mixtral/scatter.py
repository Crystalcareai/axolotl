import torch
import torch.nn as nn
from axolotl.monkeypatch.moe import GLUMLP
from torch.nn import functional as F
from transformers.activations import ACT2FN

def patch_mixtral_scatter() -> None:
    class MixtralSparseMoeBlock(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.hidden_dim = config.hidden_size
            self.ffn_dim = config.intermediate_size
            self.num_experts = config.num_local_experts
            self.top_k = config.num_experts_per_tok
            
            # gating
            self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
            self.moe_mlp = GLUMLP(
                input_size=self.hidden_dim,
                hidden_size=self.ffn_dim,
                num_experts=self.num_experts,
                top_k=self.top_k,
                activation=ACT2FN[config.hidden_act]
            )

        def forward(self, hidden_states: torch.Tensor):
            """ """
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            hidden_states = hidden_states.view(-1, hidden_dim)

            # router_logits: (batch * sequence_length, n_experts)
            router_logits = self.gate(hidden_states)
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

            # we cast back to the input dtype
            routing_weights = routing_weights.to(hidden_states.dtype)

            final_hidden_states = self.moe_mlp(hidden_states, routing_weights, selected_experts)
            final_hidden_states = final_hidden_states.view(batch_size, sequence_length, hidden_dim)
            return final_hidden_states, router_logits

    from transformers.models.mixtral import modeling_mixtral
    modeling_mixtral.MixtralSparseMoeBlock = MixtralSparseMoeBlock
    delattr(modeling_mixtral, 'MixtralBLockSparseTop2MLP')
