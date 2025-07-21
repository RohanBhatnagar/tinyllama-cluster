"""
This app turns tinyLlama into a mixture of experts model by clustering the neurons in the feedforward network into experts. 
The goal is to improve general performance of tinyllama without extra neurons. 
"""
import os
import json
import modal
import torch
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from typing import List

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("torch", "transformers", "datasets", "k_means_constrained")
)
output_volume = modal.Volume.from_name("tinyllama-experts", create_if_missing=True)
app = modal.App("emoe")

@app.function(
    image=image,
    volumes={"/outputs": output_volume},
    timeout=600,
    scaledown_window=300,
)
def run_tinyllama_inference(sample: str, idx: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

    input_ids = tokenizer.encode(sample, return_tensors="pt").to(device)
    output_ids = model.generate(input_ids, max_new_tokens=100)

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print(output_text)
    
@app.function(
    image=image,
    volumes={"/outputs": output_volume},
    timeout=600,
    scaledown_window=300,
)
def cluster_neurons(num_experts: int, layer_idx: int):
    os.makedirs(f"/outputs/layer_{layer_idx}_{num_experts}E", exist_ok=True)
    
    print(f"Clustering neurons for layer {layer_idx} into {num_experts} experts")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama_v1.1", torch_dtype=torch.float16).to(device)
    
    weights = model.model.layers[layer_idx].mlp.down_proj.weight # has shape (2048, 5632)
    intermediate_dim = weights.shape[1] # 5632
    model_dim = weights.shape[0] # 2048
    expert_size = model_dim // num_experts
    
    # normalize for clustering 
    weights = torch.nn.functional.normalize(weights, p=2, dim=-1)
    weights = weights.detach().cpu().numpy()
    
    # apply k-means constrained clustering
    kmeans = KMeansConstrained(
        n_clusters=num_experts, size_min=expert_size,
        size_max=expert_size, random_state=0, n_jobs=16,
        max_iter=1000)
    
    kmeans.fit(weights)
    
    file_name = f"/outputs/layer_{layer_idx}/experts_{num_experts}.pth"
    
    res = torch.from_nump(kmeans.labels_)
    
    with open(file_name, "wb") as f:
        torch.save(kmeans.labels_, f)
        
    return res 

    
@app.local_entrypoint()
def main():
    model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama_v1.1", torch_dtype=torch.float16)
    for idx in range(len(model.model.layers)):
        print(f"Clustering layer {idx}")
        cluster_ffns.remote(num_experts=8, layer_idx=idx)
