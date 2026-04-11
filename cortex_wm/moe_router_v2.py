import torch
import torch.nn as nn

class MOERouterV2(nn.Module):
    def __init__(self, input_dim=256, manifold_dim=128):
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(input_dim, 32) for _ in range(4)])
        self.gate = nn.Linear(input_dim, 4)

    def forward(self, x):
        weights = 0.99 * torch.softmax(self.gate(x), dim=-1) + 0.01 / 4  # unimix
        expert_outputs = [expert(x) for expert in self.experts]
        manifold = torch.cat(expert_outputs, dim=-1)
        return torch.tanh(manifold), weights
