"""
Example: How to adapt existing notebook code to use the model server.

This shows the before/after comparison.
"""

# =============================================================================
# BEFORE: Original notebook code (runs model locally)
# =============================================================================

"""
# Cell: Load model
device = 'cuda:0'
model_dir = '/data0/modelscope/Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4'
gptq_config = GPTQConfig(bits=4, group_size=128, desc_act=False, use_exllama=True, exllama_config={"version": 2})
model = AutoModelForCausalLM.from_pretrained(model_dir,
    torch_dtype=torch.float16, quantization_config=gptq_config,
    local_files_only=True, low_cpu_mem_usage=True, device_map=device)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Cell: Forward samples
for r in tqdm(_results):
    model_inputs = tokenizer([r.prompt], return_tensors="pt").to(model.device)
    set_hooks(model)
    output = model(**model_inputs, output_hidden_states=True)
    r.outputs = get_outputs(model, output)

# Cell: Get attention weights
aw = {}
for layer, head in top_heads:
    aw[(layer, head)] = torch.cat([
        get_attn_weights(model, r.outputs, layer, head, pos_ids).to('cpu')
        for r in _results
    ])
"""

# =============================================================================
# AFTER: Using model server
# =============================================================================

from model_client import ModelClient
from tqdm import tqdm
import torch

# Initialize client
client = ModelClient("http://localhost:8000")

# -----------------------------------------------------------------------------
# Step 1: Load model (only needed once, persists on server)
# -----------------------------------------------------------------------------

status = client.status()
if not status["model_loaded"]:
    print("Loading model on server...")
    client.load_model(
        model_path="/data0/modelscope/Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4",
        device="cuda:0",
        quantization="gptq"
    )
else:
    print(f"Model already loaded: {status['model_name']}")

# Get model config
L = client.config["num_layers"]
H = client.config["num_heads"]

# -----------------------------------------------------------------------------
# Step 2: Forward samples and cache activations
# -----------------------------------------------------------------------------

# Option A: Forward one by one
dataset_id = "my_puzzles"
for i, r in enumerate(tqdm(_results)):
    client.forward(dataset_id, sample_id=i, prompt=r.prompt)

# Option B: Use helper for Result objects
# client.forward_results(dataset_id, _results)

# -----------------------------------------------------------------------------
# Step 3: Get attention weights (from server cache)
# -----------------------------------------------------------------------------

pos_ids = [idx - 1 for idx in _results[0].answer_indices]  # Query positions
top_heads = [(62, 40), (62, 45), (63, 61)]  # Example heads

aw = {}
for layer, head in tqdm(top_heads, desc="Getting attn weights"):
    # Get attention weights for all samples, stacked
    aw[(layer, head)] = client.get_attn_weights_batch(
        dataset_id,
        sample_ids=list(range(len(_results))),
        layer=layer,
        head=head,
        pos_ids=pos_ids,
        device='cpu'
    )

print(f"Attention shape: {list(aw.values())[0].shape}")

# -----------------------------------------------------------------------------
# Step 4: Other operations
# -----------------------------------------------------------------------------

# Get hidden states
hs = client.get_hidden_states(dataset_id, sample_id=0, layer=60)

# Get head output
head_out = client.get_head_output(dataset_id, sample_id=0, layer=60, head=10, pos_ids=pos_ids)

# Clear cache when done
# client.clear_cache(dataset_id)

# =============================================================================
# Benefits of Server Approach
# =============================================================================

"""
1. SHARED STATE: Both notebook and Claude processes access same cached data
   - No need to reload model (16GB+ VRAM saved)
   - No need to re-run forward passes

2. PERSISTENCE: Server keeps model and cache across sessions
   - Notebook can be restarted without losing cache
   - Claude can analyze same data you're working with

3. EFFICIENCY: 
   - Model loaded once, shared by all clients
   - Activations cached in GPU/CPU memory
   - Only transfer requested tensors

4. COLLABORATION:
   - You run attribution in notebook
   - Tell Claude: "analyze dataset 'my_puzzles' with heads [(62,40), (63,61)]"
   - Claude connects to same server and gets real data!

5. FLEXIBILITY:
   - Add custom endpoints as needed
   - Extend with new analysis functions
"""

# =============================================================================
# Quick Reference: API Mapping
# =============================================================================

"""
NOTEBOOK CODE                          →  CLIENT CODE
--------------------------------------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(...)
                                       →  client.load_model(model_path, device)

set_hooks(model)
output = model(**inputs)
r.outputs = get_outputs(model, output)
                                       →  client.forward(dataset_id, sample_id, prompt)

get_attn_weights(model, r.outputs, layer, head, pos_ids)
                                       →  client.get_attn_weights(dataset_id, sample_id, layer, head, pos_ids)

get_head_output(model, outputs, layer, head, pos_ids)
                                       →  client.get_head_output(dataset_id, sample_id, layer, head, pos_ids)

get_attn_output(model, outputs, layer, pos_ids)
                                       →  client.get_attn_output(dataset_id, sample_id, layer, pos_ids)

outputs.hidden_states[layer]
                                       →  client.get_hidden_states(dataset_id, sample_id, layer)
"""

