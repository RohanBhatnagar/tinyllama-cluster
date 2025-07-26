"""
This app turns tinyLlama into a mixture of experts model by clustering the neurons in the feedforward network into experts. 
The goal is to improve general performance of tinyllama without adding any extra neurons. 
"""
import os
import modal
import warnings

warnings.filterwarnings("ignore", message=".*resume_download.*", category=FutureWarning)

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install("pip==24.0")
    .pip_install_from_requirements("requirements.txt")
)

with image.imports():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from k_means_constrained import KMeansConstrained
    import os 

experts = modal.Volume.from_name("tinyllama-experts", create_if_missing=True)
models = modal.Volume.from_name("tinyllama-models", create_if_missing=True)

app = modal.App("emoe")

@app.local_entrypoint()
def main(
    cluster: bool = False,
    rearrange: bool = False,
    num_experts: int = 8,
    num_layers: int = 22
):
    if cluster:
        layer_args = [(num_experts, idx) for idx in range(num_layers)]
        print(f"Starting parallel clustering for {num_layers} layers with {num_experts} experts each")
        results = list(cluster_neurons.starmap(layer_args))
        print(f"Completed clustering for all {num_layers} layers")
    
    if rearrange:
        print("Editing weights...")
        result = rearrange_all_neurons.remote(num_experts=num_experts)
        print(f"Rearrangement completed: {result}")
        
@app.function(
    image=image,
    volumes={"/outputs": experts, "/models": models},
    gpu="A100",
    timeout=1200, # 20 minutes 
)
def rearrange_all_neurons(num_experts: int):
    """ 
    Rearranges the model and saves the new state dict to /models/{num_experts}_Experts
    """
    from collections import OrderedDict
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama_v1.1", torch_dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama_v1.1")
    
    print("Model loaded! Beginning weight processing...")
    
    # new state dict with rearranged weights
    sd = OrderedDict()
    orig_weights = model.state_dict()
    
    for n, p in orig_weights.items():
        if 'mlp' not in n:
            sd[n] = p
            continue
        if n in sd:
            continue

        layer_idx = int(n.split('.')[2])
        pfx = '.'.join(n.split('.')[:4]) 
        
        print(f"Processing {pfx}")
        
        gate_proj_weight = orig_weights[f'{pfx}.gate_proj.weight']
        up_proj_weight = orig_weights[f'{pfx}.up_proj.weight']
        down_proj_weight = orig_weights[f'{pfx}.down_proj.weight']
        
        file_path = f"/outputs/{num_experts}_Experts/layer_{layer_idx}_{num_experts}E.pth"
        index = torch.load(file_path)
        
        sd[f'{pfx}.gate_proj.weight'] = rearrange_weights_direct(gate_proj_weight, index, num_experts, "up_or_gate")
        sd[f'{pfx}.up_proj.weight'] = rearrange_weights_direct(up_proj_weight, index, num_experts, "up_or_gate")
        sd[f'{pfx}.down_proj.weight'] = rearrange_weights_direct(down_proj_weight, index, num_experts, "down")
    
    model.load_state_dict(sd)
    print(model.state_dict().keys())
    print("Loaded rearranged weights into model!")
    
    # save the model 
    os.makedirs(f"/models/{num_experts}_Experts", exist_ok=True)
    model.save_pretrained(f"/models/{num_experts}_Experts")
    print(f"Saved model to /models/{num_experts}_Experts")
    
    # make sure i didn't kill the model 
    text = "You are a helpful assistant. Hello, how are you today?"
    input_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)
    output_ids = model.generate(input_ids, max_new_tokens=50)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    print(f"Rearranged model generated: {output_text}")
    
    return {
        "status": "success",
        "processed_layers": 22,
        "generated_text": output_text
    }

def rearrange_weights_direct(weights, index, num_experts, weight_type):
    dim1, dim2 = weights.shape
    
    print(f"Rearranging {weight_type} weights: {weights.shape}")
    
    if weight_type == "up_or_gate":
        # For gate_proj and up_proj: rearrange rows (5632 x 2048)
        expert_size = dim1 // num_experts  # 704
        new_weight = torch.zeros_like(weights)
        for i in range(num_experts):
            new_weight[i * expert_size: (i+1) * expert_size] = weights[index == i]
        return new_weight
    
    elif weight_type == "down":
        # For down_proj: transpose, rearrange, transpose back (2048 x 5632)
        expert_size = dim2 // num_experts  # 704
        new_weight = torch.zeros_like(weights.T)
        for i in range(num_experts):
            new_weight[i * expert_size: (i+1) * expert_size] = weights.T[index == i]
        return new_weight.T
    
    else:
        raise ValueError(f"Unknown weight_type: {weight_type}")

@app.function(
    image=image,
    volumes={"/outputs": experts},
    timeout=30, # 30 seconds 
)
def cluster_neurons(num_experts: int, layer_idx: int):
    os.makedirs(f"/outputs/{num_experts}_Experts", exist_ok=True)
    
    print(f"Clustering neurons for layer {layer_idx} into {num_experts} experts")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama_v1.1", torch_dtype=torch.float16).to(device)
    
    print("Model loaded!")
    
    weights = model.model.layers[layer_idx].mlp.gate_proj.weight
    intermediate_dim = weights.shape[0] # 5632
    model_dim = weights.shape[1] # 2048
    print(f"intermediate_dim: {intermediate_dim}, model_dim: {model_dim}")
    expert_size = intermediate_dim // num_experts
    
    weights = torch.nn.functional.normalize(weights, p=2, dim=-1) # normalize each row to unit length
    weights = weights.detach().cpu().numpy() # shape: (5632, 2048)
    
    kmeans = KMeansConstrained(
        n_clusters=num_experts, size_min=expert_size,
        size_max=expert_size, random_state=0, n_jobs=16,
        max_iter=1000)
    
    kmeans.fit(weights)
    
    file_name = f"/outputs/{num_experts}_Experts/layer_{layer_idx}_{num_experts}E.pth"
    res = torch.from_numpy(kmeans.labels_)
    with open(file_name, "wb") as f:
        torch.save(res, f)
        
    print(f"Saved clustering results for layer {layer_idx} to {file_name}")
    return res
