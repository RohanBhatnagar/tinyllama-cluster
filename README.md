# Doing better than LoRA fine-tuning on TinyLlama 

This experiment aims to beat baseline LoRA by only employing chunks of the feed-forward networks (FFNs) in TinyLlama-1.1B-Chat-v1.0. It does the following without adding any new params (other than LoRA): 
1. **Partitioning**: Split MLP layers into N uniform clusters based on their weight rows, as similar rows will have similar activations. 
2. **Model Surgery**: Rearrange rows so that clusters are contiguous in memory. 
3. **Routing**: Implement top-k routing during inference, sending tokens to FFN partitions whose activation is among the top-k highest partition activations. 
4. **Fine-tuning**: Finetune the modified and baseline models using LoRA. 

## Results: 

I fine-tuned TinyLlama-1.1B on Recognizing Textual Entanglement (RTE) a subset of GLUE, and observed the following: 
1. Fine-tuning the modified LLM with LoRA performs 1-2% better than the vanilla model. 
2. Modified TinyLlama achieves the accuracy of the vanilla model within 3/5 epochs and continues to increase. (N=32, k=16)
\
\
Both models were fine-tuned with **identical** settings. 
```python
LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=32,                    
    lora_alpha=64,         
    target_modules=["q_proj", "v_proj"], # Attention projections
    modules_to_save=["score"],  # Classification head (for RTE)
    bias="none",
)
```

```python
Training(
   lr=2e-5
   epochs=5
   batch_size=16 # 32 for eval
   warmup_ratio:0.1
)
```

Accuracy averaged over 5 fine-tunes for recognizing textual entanglement (RTE): 
- Original model: 71.3%
- Top-k routed: 72.9%
Before fine-tuning accuracy was between 51-55%. 

## Next Steps: 
1. Fine-tune on a diverse dataset to see if specialization can emerge. 
2. Try text to SQL. 

## Motivation 
I was interested in mixture of experts (MoE) models and obviously I don't have the resources to pretrain a model so I found this paper, hoping to find a way to unlock modularity in pretrained models. https://arxiv.org/pdf/2310.10908. I decided to explore the concepts presented in this paper by implementing them in a smaller LLM. 

