import torch
from axolotl.monkeypatch.moe import GLUMLP, ParallelExperts
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

def patch_mixtral_with_scatter_moe(model):
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, 'block_sparse_moe'):
            module = layer.block_sparse_moe
            if isinstance(module, MixtralSparseMoeBlock):
                hidden_size = module.experts[0].w2.out_features
                num_experts = len(module.experts)

                # Create new ParallelExperts for the GLUMLP
                w1w3_experts = ParallelExperts(num_experts, module.experts[0].w1.in_features, 2 * hidden_size)
                w1_weights = torch.stack([expert.w1.weight.data.to(torch.float32) for expert in module.experts])
                w3_weights = torch.stack([expert.w3.weight.data.to(torch.float32) for expert in module.experts])
                w1w3_experts.weight = torch.nn.Parameter(torch.cat([w1_weights, w3_weights], dim=-1))

                w2_experts = ParallelExperts(num_experts, hidden_size, module.experts[0].w2.out_features)
                w2_experts.weight = torch.nn.Parameter(torch.stack([expert.w2.weight.data.to(torch.float32) for expert in module.experts]))

                # Create the GLUMLP module
                smoe = GLUMLP(
                    input_size=module.experts[0].w1.in_features,
                    hidden_size=hidden_size,
                    num_experts=num_experts,
                    top_k=module.top_k,
                    activation=module.experts[0].act_fn,
                )
                smoe.experts = w1w3_experts
                smoe.output_experts = w2_experts

                # Replace the MixtralSparseMoeBlock with the GLUMLP
                setattr(model.model.layers[i], "block_sparse_moe", smoe)

                # Clean up the old module to free up memory
                old_module = module
                del old_module
                torch.cuda.empty_cache()

    torch.cuda.empty_cache()
