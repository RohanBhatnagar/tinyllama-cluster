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
    import evaluate
    import numpy as np
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
    from transformers import LlamaForSequenceClassification, LlamaTokenizer
    from datasets import load_dataset
    from peft import get_peft_model, LoraConfig, TaskType
    
""" training process 
 vanilla fine-tuning: add Lora weights to q and v proj in attention layers 
 moe fine-tuning: 
 - add lora weights where ffn is partitioned. 
 - q v proj in attention layers 
 - the router (gate proj)
"""
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
    
    model = AutoModelForSequenceClassification.from_pretrained("TinyLlama/TinyLlama_v1.1", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama_v1.1")
    model.to(device)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8, 
        lora_alpha=16, 
        target_modules=["q_proj", "v_proj"], # for baseline, target q and v proj in attention layers. 
        lora_dropout=0.05, 
        bias="none"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    def preprocess_function(examples):
        return tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True)
    
    accuracy_metric = evaluate.load("accuracy")
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy_metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir="/data/results",
        evaluation_strategy="epoch",
        learning_rate=2e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        save_total_limit=1,
        logging_dir="/data/logs",
        logging_strategy="steps",
    )
    
    dataset = load_dataset("glue", "rte")
    
    train_dataset = dataset["train"].map(preprocess_function, batched=True)
    test_dataset = dataset["test"].map(preprocess_function, batched=True)
    
    trainer = Trainer(
        training_args=training_args,
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
    
    print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
    print("Fine-tuning complete. Saving model...")
    
    model.save_pretrained("/data/models")
    tokenizer.save_pretrained("/data/models")
    print("Model saved to /data/models")
    
    