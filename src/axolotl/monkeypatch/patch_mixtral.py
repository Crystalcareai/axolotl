import torch
from axolotl.monkeypatch.moe import GLUMLP, MLP, ParallelExperts

def patch_mixtral_with_scatter_moe(model):
    for module_name, module in model.named_modules():
        if hasattr(module, 'experts'):
            if isinstance(module, MixtralSparseMoeBlock):
                hidden_size = module.experts[0].mlp.gate_proj.out_features
                new_experts = ParallelExperts(len(module.experts), module.mlp.gate_proj.in_features, 2 * hidden_size)
                new_experts.load_state_dict({f'weight': torch.stack([e.mlp.gate_proj.weight for e in module.experts] + [e.mlp.up_proj.weight for e in module.experts])})
                output_experts = ParallelExperts(len(module.experts), hidden_size, module.experts[0].mlp.down_proj.out_features)  
                output_experts.load_state_dict({f'weight': torch.stack([e.mlp.down_proj.weight for e in module.experts])})
                module.experts = GLUMLP(module.mlp.gate_proj.in_features, hidden_size, len(module.experts), module.top_k, activation=module.mlp.act_fn)
                module.experts.experts = new_experts
                module.experts.output_experts = output_experts