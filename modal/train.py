
image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install("pip==24.0")
    .pip_install_from_requirements("requirements.txt")
    .add_local_file("architecture.py", remote_path="/root/architecture.py")
    .add_local_file("train.py", remote_path="/root/train.py")
)

with image.imports():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import architecture
    
    
@app.local_entrypoint()
def main(): 


@app.function(
    image=image,
    volumes={"/models": models},
    gpu="A100",
    timeout=1200,
    num_experts: int = 8,
)
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    config = MoELlamaConfig()
    model = MoELlamaForCausalLM.from_pretrained(f"models/{num_experts}_Experts", config=config)    
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama_v1.1")
    model.to(device)
    
    
    
    