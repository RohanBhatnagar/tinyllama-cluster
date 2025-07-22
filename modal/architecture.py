import torch
import torch.nn as nn
import torch.nn.functional as F

# the router: outputs a score for each expert, we are using top-1 routing
class MoERouter(nn.Module):
    def __init__(self, model_dim, num_experts):
        super().__init__()
        self.linear = nn.Linear(model_dim, num_experts)

    def forward(self, x):
        logits = self.linear(x) 
        return torch.softmax(logits, dim=-1)
    
# the partitioned MLP, new partitioned weights are copied
class MoEMLP(nn.Module):
    def __init__(self, up_proj_weights, down_proj_weights):
        super().__init__()
        self.num_experts = up_proj_weights.shape[0]
        self.model_dim = up_proj_weights.shape[-1]

        self.up_projs = nn.ModuleList([
            nn.Linear(self.model_dim, up_proj_weights.shape[1])
            for _ in range(self.num_experts)
        ])
        self.down_projs = nn.ModuleList([
            nn.Linear(up_proj_weights.shape[1], self.model_dim)
            for _ in range(self.num_experts)
        ])

        self.router = MoERouter(self.model_dim, self.num_experts)

        for i in range(self.num_experts):
            self.up_projs[i].weight.data.copy_(up_proj_weights[i])
            self.down_projs[i].weight.data.copy_(down_proj_weights[i])

    def forward(self, x):
        scores = self.router(x) 
        top1 = scores.argmax(dim=-1)

        outputs = torch.zeros_like(x)
        for i in range(self.num_experts):
            mask = (top1 == i) 
            if mask.any():
                x_i = x[mask]
                h = F.silu(self.up_projs[i](x_i))
                y_i = self.down_projs[i](h)
                outputs[mask] = y_i

        return outputs
