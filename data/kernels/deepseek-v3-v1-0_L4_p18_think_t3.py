import torch
import torch.nn as nn
import math

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        weight_sum = self.linear.weight.sum(dim=0)
        bias_sum = self.linear.bias.sum()
        return (x @ weight_sum + bias_sum).unsqueeze(1)