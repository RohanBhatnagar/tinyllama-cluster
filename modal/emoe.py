"""
This app turns tinyLlama into a mixture of experts model by clustering the neurons in the feedforward network into experts. 
The goal is to improve general performance of tinyllama without extra neurons. 
"""
import os
import modal
import warnings

# Suppress FutureWarnings from huggingface_hub
warnings.filterwarnings("ignore", message=".*resume_download.*", category=FutureWarning)

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install("pip==24.0")
    .pip_install_from_requirements("requirements.txt")
    .add_local_file("architecture.py", remote_path="/root/architecture.py")
)

experts = modal.Volume.from_name("tinyllama-experts", create_if_missing=True)

app = modal.App("emoe")

@app.function(
    image=image,
    volumes={"/outputs": experts},
    timeout=600,
    scaledown_window=300,
)
def cluster_neurons(num_experts: int, layer_idx: int):
    import torch
    from transformers import AutoModelForCausalLM
    from k_means_constrained import KMeansConstrained

    os.makedirs(f"/outputs/{num_experts}_Experts", exist_ok=True)
    
    print(f"Clustering neurons for layer {layer_idx} into {num_experts} experts")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama_v1.1", torch_dtype=torch.float16).to(device)
    
    print("Model loaded!")
    
    # Use gate_proj for clustering (following reference implementation)
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

@app.function(
    image=image,
    volumes={"/outputs": experts}, 
    timeout=300
)
def rearrange_neurons(weights: "torch.Tensor", layer_idx: int, num_experts: int, weight_type: str = "up_or_gate"):
    """
    Rearrange weights based on clustering index
    weight_type: "up_or_gate" for gate_proj/up_proj, "down" for down_proj
    """
    import torch 
    import os 
    
    file_path = f"/outputs/{num_experts}_Experts/layer_{layer_idx}_{num_experts}E.pth"
    index = torch.load(file_path)
    print(f"✅ Loaded clustering for layer {layer_idx}: {index.shape}")
    
    dim1, dim2 = weights.shape
    print(f"Rearranging {weight_type} weights: {weights.shape}")
    
    if weight_type == "up_or_gate":
        # For gate_proj and up_proj: rearrange rows (dim1 > dim2: 5632 > 2048)
        expert_size = dim1 // num_experts  # 704
        new_weight = torch.zeros_like(weights)
        for i in range(num_experts):
            new_weight[i * expert_size: (i+1) * expert_size] = weights[index == i]
        return new_weight
    
    elif weight_type == "down":
        # For down_proj: transpose, rearrange, transpose back (dim1 < dim2: 2048 < 5632)
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
    timeout=600
)
def model_surgery(num_experts: int):
    import sys 
    sys.path.append("/root")
    from architecture import MoEMLP
    import torch
    import os
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama_v1.1", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama_v1.1") 
    print("Loaded model and tokenizer.")
    
    # Check what clustering files are available
    print("Available clustering files:")
    os.system(f"find /outputs -name '*.pth' | head -10")
    
    for idx, layer in enumerate(model.model.layers):
        print(f"Processing layer {idx}")
        
        # Get all three MLP weights
        original_gate_proj = layer.mlp.gate_proj.weight  # (5632, 2048)
        original_up_proj = layer.mlp.up_proj.weight      # (5632, 2048) 
        original_down_proj = layer.mlp.down_proj.weight  # (2048, 5632)
        
        print(f"Original shapes - gate: {original_gate_proj.shape}, up: {original_up_proj.shape}, down: {original_down_proj.shape}")
        
        # Apply same clustering index to all three weights
        new_gate_proj = rearrange_neurons.local(weights=original_gate_proj, layer_idx=idx, num_experts=num_experts, weight_type="up_or_gate")
        new_up_proj = rearrange_neurons.local(weights=original_up_proj, layer_idx=idx, num_experts=num_experts, weight_type="up_or_gate")
        new_down_proj = rearrange_neurons.local(weights=original_down_proj, layer_idx=idx, num_experts=num_experts, weight_type="down")
        
        print(f"New shapes - gate: {new_gate_proj.shape}, up: {new_up_proj.shape}, down: {new_down_proj.shape}")
        
        # Update the layer weights directly (following reference approach)
        layer.mlp.gate_proj.weight.data = new_gate_proj
        layer.mlp.up_proj.weight.data = new_up_proj  
        layer.mlp.down_proj.weight.data = new_down_proj
        
        print(f"✅ Updated layer {idx} MLP weights")
    
    print("Model surgery complete. Testing inference...")
    
    # test inference 
    text = "Hello, how are you?"
    input_ids = tokenizer.encode(text, return_tensors="pt")
    output_ids = model.generate(input_ids, max_new_tokens=50)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Generated: {output_text}")
    return output_text

@app.function(
    image=image,
    volumes={"/outputs": experts},
    timeout=300
)
def inspect_cluster_file(layer_idx: int, num_experts: int):
    import torch
    import os

    file_path = f"/outputs/{num_experts}_Experts/layer_{layer_idx}_{num_experts}E.pth"
    
    if not os.path.exists(file_path):
        print(f"[!] File not found: {file_path}")
        return

    labels = torch.load(file_path)
    
    print(f"[✓] Loaded cluster assignments for layer {layer_idx} with {num_experts} experts.")
    print(f"  - Shape: {labels.shape}")
    print(f"  - Unique expert IDs: {torch.unique(labels).tolist()}")
    print(f"  - Distribution:")
    
    for expert_id in torch.unique(labels):
        count = (labels == expert_id).sum().item()
        print(f"    Expert {expert_id.item()}: {count} neurons")

    return labels.tolist()  # Optional: return for programmatic use



@app.local_entrypoint()
def main(
    inspect: bool = False,
    rearrange: bool = False,
    cluster: bool = False,
    surgery: bool = False
):
    if cluster:
        num_layers = 22
        num_experts = 8
        
        layer_args = [(num_experts, idx) for idx in range(num_layers)]
        
        print(f"Starting parallel clustering for {num_layers} layers with {num_experts} experts each")
        
        results = list(cluster_neurons.starmap(layer_args))
        
        print(f"Completed clustering for all {num_layers} layers")
        return results 
    
    if rearrange:
        # Use remote functions only - no local model loading
        new_weights = rearrange_neurons.remote(weights=None, layer_idx=1, num_experts=8)  # This needs fixing
        model_surgery.remote(num_experts=8)
        
    if surgery: 
        # All processing happens in Modal
        result = model_surgery.remote(num_experts=8)
        print(f"Surgery result: {result}")
        
    if inspect:
        result = inspect_cluster_file.remote(layer_idx=1, num_experts=8)
        print(result)

    
