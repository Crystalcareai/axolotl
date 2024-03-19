import gc
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, TextStreamer
from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM, MixtralConfig

from axolotl.monkeypatch.patch_mixtral import patch_mixtral_with_scatter_moe

def compute_memory_used_pct(device):
    memory_used = torch.cuda.max_memory_allocated(device) / (1024**3)
    memory_pct = (
        memory_used
        / (torch.cuda.get_device_properties(device).total_memory / (1024**3))
        * 100
    )
    return memory_pct

model_path = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# Load model
config = MixtralConfig.from_pretrained(model_path, max_position_embeddings=2048, use_cache=False)
model = MixtralForCausalLM.from_pretrained(
    model_path,
    config=config,
    device_map="auto",
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
)

# Apply scatter_moe patching to the model
patch_mixtral_with_scatter_moe(model)

tokenizer = AutoTokenizer.from_pretrained(model_path)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)


# Convert prompt to tokens
prompt_template = "[INST] {prompt} [/INST]"

prompt = "You're standing on the surface of the Earth. "\
        "You walk one mile south, one mile west and one mile north. "\
        "You end up exactly where you started. Where are you?"

tokens = tokenizer(
    prompt_template.format(prompt=prompt), 
    return_tensors='pt'
).input_ids.cuda()

# Generate output
generation_output = model.generate(
    tokens, 
    streamer=streamer,
    max_new_tokens=512
)
