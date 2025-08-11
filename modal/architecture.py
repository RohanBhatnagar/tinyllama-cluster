import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaMLP
from transformers.modeling_utils import PreTrainedModel
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaMLP,
    LlamaModel,
    LlamaRMSNorm,
    LlamaForCausalLM
)

class MoELlamaConfig(LlamaConfig):
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,                    
        hidden_size=2048,                    
        intermediate_size=5632,             
        num_hidden_layers=22,              
        num_attention_heads=8,              
        num_key_value_heads=None,      
        hidden_act="silu",                  
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,

        # EMoE-specific settings
        split_start_layer=10, # only partition last half 
        split_every_layer=2, # partition every other layer 
        topk=1,                             
        n_expert=8,                         
        mode='EMoE',                         
        select='gate',
        **kwargs,
    ):
        super().__init__(
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            hidden_act,
            max_position_embeddings,
            initializer_range,
            rms_norm_eps,
            use_cache,
            pad_token_id,
            bos_token_id,
            eos_token_id,
            pretraining_tp,
            tie_word_embeddings,
            rope_theta,
            rope_scaling,
            **kwargs,
        )
        self.split_start_layer = split_start_layer
        self.split_every_layer = split_every_layer
        self.topk = topk
        self.n_expert = n_expert
        self.select = select
        self.mode = mode


# the partitioned MLP with top-1 gating
class EMoELlamaMLP(LlamaMLP):
    def __init__(self, config):
        super().__init__(config)
        self.config = config # clustering FFNs weights to construct the experts
        
    def forward(self, x):
        # in gate proj, calculate the avg of activations in each expert, then select the top-k=1 expert for each token
        gate_output = self.gate_proj(x)
        intermediate_states = self.act_fn(gate_output) * self.up_proj(x) 
        expert_scores = gate_output.reshape(x.shape[0], x.shape[1], self.config.n_expert, -1)
        expert_scores = torch.mean(expert_scores, dim=-1)  # (bs, seq_len, n_expert)
        expert_top1_indices = torch.topk(expert_scores, k=1, dim=-1).indices  # (bs, seq_len, 1)
        expert_top1_mask = torch.zeros_like(expert_scores)  # (bs, seq_len, n_expert)
        expert_top1_mask.scatter_(dim=-1, index=expert_top1_indices, value=1)
        
        expert_top1_mask = expert_top1_mask.repeat_interleave(
            self.config.intermediate_size // self.config.n_expert, dim=-1)  # (bs, seq_len, intermediate_size)
        intermediate_states = intermediate_states * expert_top1_mask
        
        down_proj = self.down_proj(intermediate_states)
        return down_proj
            
class MoELlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: MoELlamaConfig, l_idx: int):
        super().__init__(config)
        # replace the default MLP with EMoE one when configured
        if config.mode == "EMoE" \
           and l_idx >= config.split_start_layer \
           and (l_idx - config.split_start_layer) % config.split_every_layer == 0:
            self.mlp = EMoELlamaMLP(config)
        else:
            # fall back to the original LlamaMLP
            self.mlp = LlamaMLP(config)


class MoELlamadModel(LlamaModel):
    def __init__(self, config: MoELlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        # now every layer will pick the right MLP
        self.layers = nn.ModuleList(
            [MoELlamaDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_init()


class MoELlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config: MoELlamaConfig):
        super().__init__(config)
        self.model = MoELlamadModel(config)
