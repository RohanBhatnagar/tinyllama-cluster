"""
Common Training Script for TinyLlama Baseline and Partitioned Models
Supports both baseline LoRA fine-tuning and modified model fine-tuning on RTE dataset.
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
    .add_local_file("../surgery.py", remote_path="/root/surgery.py")
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
    from architecture import ClusteredLlamaConfig, ClusteredLlamaForSequenceClassification

@app.local_entrypoint()
def main(
    mode: str = "baseline",  # "baseline", "clustered", or "compare"
    num_experts: int = 8,
    topk: int = 1,
    compare_original: bool = True
):    
    results = {}
    
    if mode in ["baseline", "compare"]:
        baseline_result = train_baseline.remote()
        results["baseline"] = baseline_result
    
    if mode in ["clustered", "compare"]:
        clustered_result = train_clustered.remote(num_experts, topk)
        results["clustered"] = clustered_result
    
    for model_type, result in results.items():
        if result:
            accuracy = result.get('eval_accuracy', 'N/A')
            print(f"{model_type.upper():>12}: {accuracy:.4f}" if isinstance(accuracy, float) else f"{model_type.upper():>12}: {accuracy}")
    
    return results

models = modal.Volume.from_name("tinyllama-models", create_if_missing=True)
data = modal.Volume.from_name("tinyllama-data", create_if_missing=True)

def get_shared_training_args(output_dir: str):
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

def get_shared_lora_config():
    return LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj"],
        modules_to_save=["score"],
        bias="none"
    )

def setup_tokenizer_and_data():
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading original TinyLlama model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading baseline TinyLlama model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
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
def train_clustered(num_experts: int = 8, topk: int = 1):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load clustered model
    try:
        model_path = f"/models/{num_experts}_Experts"
        print(f"Loading clustered model from: {model_path}")
        config = ClusteredLlamaConfig.from_pretrained(model_path)
        model = ClusteredLlamaForSequenceClassification.from_pretrained(model_path, config=config)
        print(f"Successfully loaded clustered model with {config.n_expert} experts, top-{config.topk} routing")
    except Exception as e:
        print(f"Error loading clustered model: {e}")
        print(f"Make sure you have run the clustered conversion first to create {model_path}")
        return None
    
    tokenizer, train_dataset, eval_dataset = setup_tokenizer_and_data()
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    
    # Apply LoRA
    model = get_peft_model(model, get_shared_lora_config())
    model.print_trainable_parameters()
    
    trainer = Trainer(
        args=get_shared_training_args("/data/clustered_results"),
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=get_compute_metrics(),
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    
    print("Starting clustered LoRA fine-tuning...")
    trainer.train()
    
    eval_results = trainer.evaluate()
    print(f"Clustered fine-tuned accuracy: {eval_results['eval_accuracy']:.4f}")
    
    # Save model
    os.makedirs("/models/clustered_finetuned", exist_ok=True)
    model.save_pretrained("/models/clustered_finetuned")
    tokenizer.save_pretrained("/models/clustered_finetuned")
    print("Model saved.")
    
    return eval_results 