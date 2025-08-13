"""
Unified Training Script for TinyLlama Baseline and eMoE Models
Supports both baseline LoRA fine-tuning and eMoE model fine-tuning on RTE dataset.
"""

import os
import modal
import warnings
from typing import Optional, Dict, Any

warnings.filterwarnings("ignore", message=".*resume_download.*", category=FutureWarning)

app = modal.App(name="unified-tinyllama-training")

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install("pip==24.0")
    .pip_install_from_requirements("requirements.txt")
    .add_local_file("../architecture.py", remote_path="/root/architecture.py")
    .add_local_file("../emoe.py", remote_path="/root/emoe.py")
)

with image.imports():
    import torch
    import numpy as np
    from transformers import (
        AutoModelForSequenceClassification, 
        AutoTokenizer, 
        DataCollatorWithPadding,
        Trainer, 
        TrainingArguments
    )
    from datasets import load_dataset
    import evaluate
    from peft import get_peft_model, LoraConfig, TaskType
    from architecture import MoELlamaConfig, EMoELlamaForSequenceClassification

@app.local_entrypoint()
def main(
    mode: str = "baseline",  # "baseline", "emoe", or "compare"
    num_experts: int = 8,
    topk: int = 1,
    compare_original: bool = True
):
    """
    Main entry point for training experiments
    
    Args:
        mode: "baseline" for standard LoRA, "emoe" for eMoE model, "compare" for both
        num_experts: Number of experts for eMoE mode
        topk: Top-k routing for eMoE mode  
        compare_original: Whether to evaluate original model first
    """
    print(f"🚀 Starting {mode.upper()} training experiment...")
    
    results = {}
    
    if mode == "compare" or compare_original:
        print("\n" + "="*60)
        print("EVALUATING ORIGINAL MODEL")
        print("="*60)
        original_result = evaluate_original.remote()
        results["original"] = original_result
        print(f"Original model results: {original_result}")
    
    if mode in ["baseline", "compare"]:
        print("\n" + "="*60)
        print("TRAINING BASELINE MODEL")
        print("="*60)
        baseline_result = train_baseline.remote()
        results["baseline"] = baseline_result
        print(f"Baseline results: {baseline_result}")
    
    if mode in ["emoe", "compare"]:
        print("\n" + "="*60)
        print("TRAINING eMoE MODEL")
        print("="*60)
        emoe_result = train_emoe.remote(num_experts, topk)
        results["emoe"] = emoe_result
        print(f"eMoE results: {emoe_result}")
    
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    for model_type, result in results.items():
        if result:
            accuracy = result.get('eval_accuracy', 'N/A')
            print(f"{model_type.upper():>12}: {accuracy:.4f}" if isinstance(accuracy, float) else f"{model_type.upper():>12}: {accuracy}")
    print("="*60)
    
    return results

models = modal.Volume.from_name("tinyllama-models", create_if_missing=True)
data = modal.Volume.from_name("tinyllama-data", create_if_missing=True)

# Shared configuration and helper functions
def get_shared_training_args(output_dir: str) -> TrainingArguments:
    """Get standard training arguments used by both baseline and eMoE"""
    return TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        weight_decay=0.0,
        warmup_ratio=0.1,
        save_strategy="epoch",
        seed=42,
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        report_to=None,
    )

def get_shared_lora_config() -> LoraConfig:
    """Get standard LoRA configuration used by both models"""
    return LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj"],
        modules_to_save=["score"],
        bias="none"
    )

def setup_tokenizer_and_data():
    """Setup tokenizer and dataset preprocessing - shared by all models"""
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    def preprocess(examples):
        return tokenizer(examples["sentence1"], examples["sentence2"], 
                        truncation=True, max_length=512, padding=False)
    
    ds = load_dataset("glue", "rte")
    train_dataset = ds["train"].map(preprocess, batched=True).rename_column("label", "labels")
    eval_dataset = ds["validation"].map(preprocess, batched=True).rename_column("label", "labels")
    
    return tokenizer, train_dataset, eval_dataset

def get_compute_metrics():
    """Get metrics computation function - shared by all models"""
    accuracy_metric = evaluate.load("accuracy")
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy_metric.compute(predictions=predictions, references=labels)
    
    return compute_metrics

@app.function(
    image=image,
    volumes={"/models": models, "/data": data},
    gpu="A100",
    timeout=7200,
)
def evaluate_original():
    """Evaluate the original TinyLlama model on RTE validation set"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading original TinyLlama model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
        torch_dtype=torch.float16, 
        num_labels=2
    )
    
    tokenizer, _, eval_dataset = setup_tokenizer_and_data()
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    
    trainer = Trainer(
        model=model,
        eval_dataset=eval_dataset,
        compute_metrics=get_compute_metrics(),
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    
    print("Evaluating original model...")
    eval_results = trainer.evaluate()
    print(f"Original model accuracy: {eval_results['eval_accuracy']:.4f}")
    
    return eval_results

@app.function(
    image=image,
    volumes={"/models": models, "/data": data},
    gpu="A100",
    timeout=7200,
)
def train_baseline():
    """Train baseline TinyLlama with LoRA on RTE"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading baseline TinyLlama model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
        torch_dtype=torch.float16, 
        num_labels=2
    )
    
    tokenizer, train_dataset, eval_dataset = setup_tokenizer_and_data()
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    
    # Apply LoRA
    model = get_peft_model(model, get_shared_lora_config())
    model.print_trainable_parameters()
    
    print("\n" + "="*50)
    print("BASELINE MODEL ARCHITECTURE")
    print("="*50)
    print("Model: TinyLlama-1.1B-Chat + LoRA")
    print("LoRA rank: 32, alpha: 64")
    print("Target modules: q_proj, v_proj")
    print("="*50)
    
    trainer = Trainer(
        args=get_shared_training_args("/data/baseline_results"),
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=get_compute_metrics(),
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    
    print("Starting baseline LoRA fine-tuning...")
    trainer.train()
    
    eval_results = trainer.evaluate()
    print(f"Baseline fine-tuned accuracy: {eval_results['eval_accuracy']:.4f}")
    
    # Save model
    os.makedirs("/models/baseline_finetuned", exist_ok=True)
    model.save_pretrained("/models/baseline_finetuned")
    tokenizer.save_pretrained("/models/baseline_finetuned")
    print("Baseline model saved to /models/baseline_finetuned")
    
    return eval_results

@app.function(
    image=image,
    volumes={"/models": models, "/data": data},
    gpu="A100",
    timeout=7200,
)
def train_emoe(num_experts: int = 8, topk: int = 1):
    """Train eMoE TinyLlama with LoRA on RTE"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load eMoE model
    try:
        model_path = f"/models/{num_experts}_Experts"
        print(f"Loading eMoE model from: {model_path}")
        config = MoELlamaConfig.from_pretrained(model_path)
        model = EMoELlamaForSequenceClassification.from_pretrained(model_path, config=config)
        print(f"Successfully loaded eMoE model with {config.n_expert} experts, top-{config.topk} routing")
    except Exception as e:
        print(f"Error loading eMoE model: {e}")
        print(f"Make sure you have run the eMoE conversion first to create {model_path}")
        return None
    
    tokenizer, train_dataset, eval_dataset = setup_tokenizer_and_data()
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    
    # Apply LoRA
    model = get_peft_model(model, get_shared_lora_config())
    model.print_trainable_parameters()
    
    # Print eMoE architecture info
    print("\n" + "="*50)
    print("eMoE MODEL ARCHITECTURE")
    print("="*50)
    expert_stats = model.get_expert_utilization_stats()
    emoe_layers = [k for k, v in expert_stats.items() if v.get('is_emoe', False)]
    print(f"Total layers: {len(expert_stats)}")
    print(f"eMoE layers: {len(emoe_layers)}")
    print(f"Dense layers: {len(expert_stats) - len(emoe_layers)}")
    print(f"Experts per eMoE layer: {config.n_expert}")
    print(f"Top-k routing: {config.topk}")
    print("LoRA rank: 32, alpha: 64")
    print("Target modules: q_proj, v_proj")
    print("="*50)
    
    trainer = Trainer(
        args=get_shared_training_args("/data/emoe_results"),
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=get_compute_metrics(),
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    
    print("Starting eMoE LoRA fine-tuning...")
    trainer.train()
    
    eval_results = trainer.evaluate()
    print(f"eMoE fine-tuned accuracy: {eval_results['eval_accuracy']:.4f}")
    
    # Save model
    os.makedirs("/models/emoe_finetuned", exist_ok=True)
    model.save_pretrained("/models/emoe_finetuned")
    tokenizer.save_pretrained("/models/emoe_finetuned")
    print("eMoE model saved to /models/emoe_finetuned")
    
    # Expert analysis
    print("\n" + "="*50)
    print("TRAINING COMPLETED - EXPERT ANALYSIS")
    print("="*50)
    final_stats = model.get_expert_utilization_stats()
    print(f"Final model has {len([v for v in final_stats.values() if v.get('is_emoe', False)])} eMoE layers")
    print("="*50)
    
    return eval_results 