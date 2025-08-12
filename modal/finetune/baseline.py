"""
Finetuning TinyLlama-1.1 on RTE dataset under baseline settings. 
Added LoRA weights to q and v projections in attention layers. 
"""

import os
import modal
import warnings 

warnings.filterwarnings("ignore", message=".*resume_download.*", category=FutureWarning)

app = modal.App(name="train-emoe")

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install("pip==24.0")
    .pip_install_from_requirements("requirements.txt")
)

with image.imports():
    import torch
    import numpy as np
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
    from datasets import load_dataset
    import evaluate
    from transformers import Trainer, TrainingArguments
    from peft import get_peft_model, LoraConfig, TaskType
    
""" 
training process 
- vanilla fine-tuning: add Lora weights to q and v proj in attention layers 
"""
@app.local_entrypoint()
def main(finetune: bool = False, baseline: bool = False): 
    if baseline: 
        print("Running original model evaluation...")
        original_results = evaluate_original.remote()
        print(f"Original model results: {original_results}")
    
    if finetune: 
        print("Running fine-tuned model training...")
        train.remote()
    
models = modal.Volume.from_name("tinyllama-models", create_if_missing=True)
data = modal.Volume.from_name("tinyllama-data", create_if_missing=True)

@app.function(
    image=image,
    volumes={"/models": models, "/data": data},
    gpu="A100",
    timeout=7200, # 2 hours
)
def evaluate_original():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained("TinyLlama/TinyLlama_v1.1", num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama_v1.1")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    
    model.to(device)
    
    def preprocess_function(examples):
        return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)
    
    accuracy_metric = evaluate.load("accuracy")
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy_metric.compute(predictions=predictions, references=labels)
    
    # Load and preprocess dataset
    dataset = load_dataset("glue", "rte")
    eval_dataset = dataset["validation"].map(preprocess_function, batched=True)
    
    # Create trainer for evaluation only
    trainer = Trainer(
        model=model,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    
    print("Evaluating original model on RTE validation set...")
    eval_results = trainer.evaluate()
    
    print(f"Original model accuracy: {eval_results['eval_accuracy']:.4f}")
    return eval_results
    
@app.function(
    image=image,
    volumes={"/models": models, "/data": data},
    gpu="A100",
    timeout=7200, # 2 hours
)
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = AutoModelForSequenceClassification.from_pretrained("TinyLlama/TinyLlama_v1.1", num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama_v1.1", use_fast=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id # set pad token id for the model to match the tokenizer
    
    model.to(device)
    
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=32, lora_alpha=64,
        target_modules=["q_proj", "v_proj"],
        modules_to_save=["score"],   
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
        
    def preprocess(examples):
        return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)

    ds = load_dataset("glue","rte")
    tokenized_ds = ds.map(preprocess, batched=True)
    tokenized_ds = tokenized_ds.rename_column("label","labels")
    tokenized_ds = tokenized_ds.remove_columns(["sentence1", "sentence2", "idx"])
    
    train_dataset = tokenized_ds["train"]
    test_dataset = tokenized_ds["validation"]
    
    accuracy_metric = evaluate.load("accuracy")
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy_metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir="/data/results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_ratio=0.1,
        weight_decay=0.0,
        seed=42,
    )
    
    trainer = Trainer(
        args=training_args,
        model=model, 
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    
    print("Starting fine-tune on RTE under baseline settings.")
    trainer.train() 
    
    print("Fine-tune completed. Final evaluation results:")
    eval_results = trainer.evaluate()
    
    print(f"Fine-tuned model accuracy: {eval_results['eval_accuracy']:.4f}")
    print("Fine-tuning complete. Saving model...")
    
    os.makedirs("/models/baseline", exist_ok=True)
    model.save_pretrained("/models/baseline")
    tokenizer.save_pretrained("/models/baseline")
    
    print("Model saved to /models/baseline")
    return eval_results
    
    