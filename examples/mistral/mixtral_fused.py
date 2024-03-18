import torch
from axolotl.monkeypatch.axolotl_mixtral_moe_patch import patch_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

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
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", load_in_4bit=False)
patch_model(model)

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
