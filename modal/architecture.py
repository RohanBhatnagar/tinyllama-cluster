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
    LlamaForCausalLM,
    LlamaForSequenceClassification
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
        topk=4,                             
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

# the partitioned MLP with top-k gating
class EMoELlamaMLP(LlamaMLP):
    def __init__(self, config):
        super().__init__(config)
        self.config = config # clustering FFNs weights to construct the experts
        self.k = self.config.topk
        
        up = self.up_proj.weight.detach().contiguous()
        int_dim, hidden_dim = up.size()
        s = self.config.intermediate_size // self.config.n_expert
        self.s = s 
        # the mean of each expert's weights 
        G = up.view(self.config.n_expert, s, hidden_dim).mean(dim=1).T
        # G will move to device as a buffer 
        self.register_buffer("G", G, persistent=False)

        
    def forward(self, x):
        # calculate gate_proj and up_proj, then take elementwise product 
        # (that's just how swiglu works, unfortunate)
        u = F.linear(x, self.up_proj.weight)
        g = self.act_fn(F.linear(x, self.gate_proj.weight))
        h = u * g
        scores = torch.matmul(x, self.G)
        topk_idx = scores.topk(self.k, dim=-1).indices
        sel = torch.zeros_like(scores).scatter(-1, topk_idx, 1.0)
        token_mask = sel.repeat_interleave(self.s, dim=-1)
        h = h * token_mask.type_as(h)
        y = F.linear(h, self.down_proj.weight)
        return y
            
class MoELlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: MoELlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        if config.mode == "EMoE" \
           and layer_idx >= config.split_start_layer \
           and (layer_idx - config.split_start_layer) % config.split_every_layer == 0:
            self.mlp = EMoELlamaMLP(config)
        else:
            self.mlp = LlamaMLP(config)


class MoELlamaModel(LlamaModel):
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
        self.model = MoELlamaModel(config)


class EMoELlamaForSequenceClassification(LlamaForSequenceClassification):
    """eMoE Llama model for sequence classification tasks"""
    
    def __init__(self, config: MoELlamaConfig):
        super().__init__(config)
        self.model = MoELlamaModel(config)
        self.score = nn.Linear(config.hidden_size, config.num_labels, bias=False)
        self.post_init()
    
    def get_expert_utilization_stats(self):
        """Return statistics about expert usage across eMoE layers"""
        stats = {}
        for i, layer in enumerate(self.model.layers):
            if hasattr(layer.mlp, 'config') and layer.mlp.config.mode == 'EMoE':
                stats[f'layer_{i}'] = {
                    'is_emoe': True,
                    'num_experts': layer.mlp.config.n_expert
                }
            else:
                stats[f'layer_{i}'] = {'is_emoe': False}
        return stats
