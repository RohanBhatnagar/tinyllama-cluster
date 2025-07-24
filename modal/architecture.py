import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaMLP

# the partitioned MLP with top-2 gating
class EMoELlamaMLP(LlamaMLP):
    def __init__(self, config):
        super().__init__(config)
        self.config = config # clustering FFNs weights to construct the experts
        
    def forward(self, x):
        gate_output = self.gate_proj(x)
        intermediate_states = self.act_fn(gate_output) * self.up_proj(x) 
        
        # Reshape to get expert scores: (bs, seq_len, n_expert, expert_size)
        expert_scores = gate_output.reshape(x.shape[0], x.shape[1], self.config.n_expert, -1)
        
        # Avg-k gating: average the activation scores across expert dimensions
        expert_scores = torch.mean(expert_scores, dim=-1)  # (bs, seq_len, n_expert)
        
        # Top-2 gating: select top 2 experts for each token
        expert_top2_indices = torch.topk(expert_scores, k=2, dim=-1).indices  # (bs, seq_len, 2)
        
        # Create mask for top-2 experts only
        expert_top2_mask = torch.zeros_like(expert_scores)  # (bs, seq_len, n_expert)
        expert_top2_mask.scatter_(dim=-1, index=expert_top2_indices, value=1)
        
        # Expand mask to match intermediate_states dimensions
        expert_top2_mask = expert_top2_mask.repeat_interleave(
            self.config.intermediate_size // self.config.n_expert, dim=-1)  # (bs, seq_len, intermediate_size)
        
        # Apply top-2 mask: only activate neurons from top-2 experts
        intermediate_states = intermediate_states * expert_top2_mask
        
        # Final projection
        down_proj = self.down_proj(intermediate_states)
        return down_proj