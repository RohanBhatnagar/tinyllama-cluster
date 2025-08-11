import modal
import warnings 

warnings.filterwarnings("ignore", message=".*resume_download.*", category=FutureWarning)

app = modal.App(name="train-emoe")

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
    from architecture import MoELlamaConfig, MoELlamaForCausalLM
    from transformers import LlamaForCausalLM, LlamaTokenizer
    from datasets import load_dataset

models = modal.Volume.from_name("tinyllama-models")
    
    
""" training process """
@app.local_entrypoint()
def main(): 
    train.remote()
    


@app.function(
    image=image,
    volumes={"/models": models},
    gpu="A100",
    timeout=1200,
)
def train(
    num_experts: int = 8,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    config = MoELlamaConfig(
        n_expert=num_experts,
        num_attention_heads=32,
        num_key_value_heads=4
    )
    model = MoELlamaForCausalLM(config)
    
    model_path = f"/models/{num_experts}_Experts"
    model.load_state_dict(torch.load(f"{model_path}/pytorch_model.bin", map_location=device))
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama_v1.1")
    model.to(device)
    
    input = tokenizer("Who was the first president of the United States?", return_tensors="pt").to(device)
    output = model.generate(**input, max_new_tokens=100)
    print(tokenizer.decode(output[0], skip_special_tokens=True))


@app.function(
    image=image,
    volumes={"/models": models},
    gpu="A100",
    timeout=1200,
)
def reference(
    num_experts: int = 8,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LlamaForCausalLM.from_pretrained("TinyLlama/TinyLlama_v1.1", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama_v1.1")
    model.to(device)

    input = tokenizer("Who was the first president of the United States?", return_tensors="pt").to(device)
    output = model.generate(
        **input, 
        max_new_tokens=100,
    )
    print(tokenizer.decode(output[0], skip_special_tokens=True))
    
    